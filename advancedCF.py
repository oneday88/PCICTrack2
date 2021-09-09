import os, logging
import _pickle as pickle

import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn

from datetime import datetime
"""
Load the dataset
"""
### Load the dataset
with open('trainPrepare.pkl', 'rb') as f:
    [totalDt,validDt, rating] = pickle.load(f)
#rating.loc[rating.rating==5, 'rating']  = 4
#rating.loc[rating.rating==1, 'rating']  = 2
"""
augument = pd.read_csv("augument.csv")
augument['label'] = 1
augument.loc[augument['pred']<0.5,'label'] = 0
augument = augument.loc[((augument.pred<0.1) | (augument.pred>0.9)),]
"""

"""
The data for model training
"""
trainFeatures = ["userid", "tagid"]
#debiasFeatures = ["userid", "tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7","tag8","movieAveRate"]
trainX = totalDt[trainFeatures].values
trainY = totalDt['label'].values
validX = validDt[trainFeatures].values
validY = validDt['label'].values


##The weighting for the loss function
trainX = np.array(trainX[:,0:2], dtype='f')
validX = np.array(validX, dtype='f')
trainY = np.array(trainY, dtype='f')
validY = np.array(validY, dtype='f')

logging.basicConfig(level=logging.DEBUG, filename="advanceCF.log", filemode="w",format="%(asctime)-15s %(levelname)-8s %(message)s")

"""
The baseline CF model
"""
embeddingSize = 10
dataCtx = mx.cpu()
modelCtx = mx.cpu()

maxUserCount = int(trainX[:,0].max())+1
maxTagCount = int(trainX[:,1].max())+1
print(maxUserCount)
print(maxTagCount)

from CFModels import advancedCF
from CFTrainer2 import CFTrainer
pmfModel = advancedCF(embeddingSize, maxUserCount, maxTagCount)
cf = CFTrainer(pmfModel,  dataCtx, modelCtx)
###The parameters for deep learning framework
learningRate = 0.0001
batchSize = 64
epochs = 50

### Parameters for the model

CELoss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
huberLoss = gluon.loss.HuberLoss()
initializer = mx.init.Xavier(magnitude=2.24)
#initializer = mx.initializer.Orthogonal()
optimizer = 'adam';

trainerParamsList = {'epochs': epochs, 'batchSize': batchSize, 'esEpochs': 4, 'learningRate': learningRate,
        'lossFunc': huberLoss, 'initializer': initializer, 'optimizer':optimizer, "ratingReg": 0.5, "isSampleTag": 0, 'isAddConstant':0}
"""
The model training
"""
mark ='debiasMF'
trainingLog = cf.fit(mark, trainX, trainY, rating, validX, validY, trainerParamsList)

