import math,os,random
import logging
from collections import deque

import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn
from tqdm import tqdm

from metrics import MAE, MSE, RMSE, AUC
"""
The trainer framework for different CF (collaborative filtering) models
"""
class CFTrainer(object):
    def __init__(self, MFModel=None, modelCtx=mx.cpu(), dataCtx=mx.cpu()):
        self.modelCtx = modelCtx
        self.dataCtx = dataCtx
        self.model = MFModel
    
    def saveModel(self,nnModel, e, pathDir, mark, metric):
        if not os.path.exists(pathDir):
            os.makedirs(pathDir)
        filename = os.path.join(pathDir, "{}_epoch_{}_metrics_{:.3f}".format(mark, e, metric))
        filename += '.param'
        nnModel.save_parameters(filename)
        return filename

    def Evaluator(self, testX, testY):
        testX = nd.array(testX, ctx=self.dataCtx)
        true = testY
        preds = self.model.predict(testX).asnumpy()
        preds[preds<0]= 0
        preds[preds>1] = 1

        mae = MAE(preds, true)
        mse = MSE(preds, true)
        rmse = RMSE(preds, true)
        auc = AUC(true, preds)
        return preds, mae, mse, rmse, auc

    def getSubmit(self, testDt):
        testX = nd.array(testDt[:, :2], ctx=self.dataCtx)
        preds = self.model.predict(testX).asnumpy()
        preds[preds<0]= 0
        preds[preds>1] = 1
        submit= np.hstack([testDt, preds.reshape(-1,1)])
        return submit

    def fit(self, mark, trainX, trainY, ratingDt, validX, validY,  paramsDict):
        ### The parameters
        epochs = paramsDict['epochs']
        esEpochs = paramsDict['esEpochs']
        # The sample rate
        batchSize = paramsDict['batchSize']
        learningRate = paramsDict['learningRate']
        # The sample rate

        optimizer = paramsDict['optimizer']
        initializer = paramsDict['initializer']
        lossFunc = paramsDict['lossFunc']
        ratingReg = paramsDict['ratingReg']
        isSampleTag = paramsDict['isSampleTag']
        isAddConstant = paramsDict['isAddConstant']
        ### The model initialization
        self.model.collect_params().initialize(initializer, ctx=self.modelCtx,force_reinit=True)
        ### The trainer
        trainer = gluon.Trainer(self.model.collect_params(), optimizer=optimizer, optimizer_params={'learning_rate': learningRate})

        debiasFeatures = ["userid", "tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7","tag8","movieAveRate"]
        ##The training
        numSamples = len(trainY)
        batchCount = int(math.ceil(len(trainY) / batchSize))
        

        bestValidMetric = 0

        history= {}
        modelDeque= deque()
        trainLossSeq = []
        testAUCSeq = []
        for e in tqdm(range(epochs), desc='epochs'):
            print(bestValidMetric)
            debiasFeatures = ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7","tag8"]
            if isSampleTag:
                debiasFeatures = ["userid"]+random.sample(debiasFeatures, 5)+['movieAveRate']
            else:
                debiasFeatures = ["userid"]+debiasFeatures+['movieAveRate']
            print(debiasFeatures)
            debiasDt = ratingDt.sample(n=numSamples, replace=True)
            debiasX = np.array(debiasDt[debiasFeatures].values,dtype='f')
            debiasY = np.array(debiasDt['rating'].values, dtype='f')

            trainData = gluon.data.DataLoader(gluon.data.ArrayDataset(trainX, trainY,debiasX, debiasY), batch_size=batchSize, shuffle=True)
            cumulativeLoss = 0
            for i, (data, label, subX, subY) in enumerate(trainData):
                data = data.as_in_context(self.dataCtx)
                label = label.as_in_context(self.dataCtx)
                subX = subX.as_in_context(self.dataCtx)
                subY = subY.as_in_context(self.dataCtx)
                with autograd.record():
                    output, output2 = self.model(data, subX)
                    loss = lossFunc(output, label)
                    loss2 = lossFunc(output2, subY)
                    loss = loss+loss2* ratingReg
                loss.backward()
                trainer.step(batchSize)
                batchLoss = nd.sum(loss).asscalar()
                batchAvgLoss = batchLoss / data.shape[0]
                cumulativeLoss += batchLoss
            #logging.info("Epoch %s / %s, Batch %s / %s. Loss: %s" % (e + 1, epochs, i + 1, batchCount, batchAvgLoss))
            trainPreds, trainMae, trainMse, trainRmse, trainAuc = self.Evaluator(trainX, trainY)
            logging.info("Epoch %s / %s. Loss: %s. " % (e + 1, epochs, cumulativeLoss / numSamples))
            logging.info("Epoch %s / %s. train AUC: %.6f. trainMse: %.6f. trainMae: %.6f. trainRmse: %.6f.  maxPred %.6f, minPred %.6f" % (e + 1, epochs, trainAuc,trainMse,trainMae, trainRmse, max(trainPreds), min(trainPreds)))
            trainLossSeq.append(cumulativeLoss)
            if validX is None:
                pass
            else: 
                preds, testMae, testMse, testRmse, testAuc = self.Evaluator(validX, validY)
                logging.info("Epoch %s / %s. test AUC: %.6f. testMse: %.6f. testMae: %.6f. testRmse: %.6f.  maxPred %.6f, minPred %.6f" % (e + 1, epochs, testAuc,testMse,testMae, testRmse, max(preds), min(preds)))
                testAUCSeq.append(testAuc)
                saveEpoch = 15
                if((e>=saveEpoch) & (testAuc > bestValidMetric)):
                    tmpModel = self.saveModel(self.model, e, 'Params', mark, testAuc)
                    modelDeque.clear()
                    modelDeque.append(tmpModel)
                    bestValidMetric = testAuc

                elif ((e>=saveEpoch) & (len(modelDeque)>0) and (len(modelDeque) < esEpochs)):
                    modelDeque.append(tmpModel)
                elif ((e>=saveEpoch) & (len(modelDeque)>0)):
                    break

        history['trainLoss'] = trainLossSeq
        history['validAUC'] = testAUCSeq
        bestModel = modelDeque.popleft()
        return history,bestModel,bestValidMetric
