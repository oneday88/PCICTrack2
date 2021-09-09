import os, logging,re
import _pickle as pickle

import numpy as np
import pandas as pd

import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn

from datetime import datetime


"""
The submit dataset
"""
test = np.loadtxt('./test_phase2.txt', dtype=int)
"""
The baseline CF model
"""
from CFModels import advancedCF
from CFTrainer import CFTrainer

dataCtx = mx.cpu()
modelCtx = mx.cpu()

maxUserCount = 1000
maxTagCount = 1720

"""
The base debias model
"""

paramFileList = \
        ["Params/huber_15_1.3_0_0_epoch_18_metrics_0.805.param",
          "Params/huber_15_1.3_0_0_epoch_16_metrics_0.804.param",
          "Params/huber_15_1.5_0_0_epoch_18_metrics_0.802.param",
          "Params/huber_15_1.5_0_0_epoch_17_metrics_0.802.param"]
submitList = []
for paramFile in paramFileList:
    paramFile = paramFile[:-6]
    paramList = re.split("_|/",paramFile)
    embeddingSize = int(paramList[2])
    isSampleTag = int(paramList[4])
    isAddConstant = int(paramList[5])
    validAUC = float(paramList[9])
    pmfModel = advancedCF(embeddingSize, maxUserCount,maxTagCount, isSampleTag, isAddConstant)
    pmfModel.load_parameters(paramFile+".param", ctx=modelCtx)
    cf = CFTrainer(pmfModel,  dataCtx, modelCtx)
    submit = cf.getSubmit(test)[:,2:3]
    submitList.append(submit)

ensembleSubmit = np.hstack(submitList).mean(axis=1)

"""
The IPSM debias model
"""
paramFileList = [
 'Params/Bothhuber_15_6.0_0_1_epoch_14_metrics_0.806.param',
 'Params/Bothhuber_15_6.0_0_1_epoch_15_metrics_0.806.param',
 'Params/Bothhuber_15_6.0_0_1_epoch_16_metrics_0.806.param',
 'Params/Bothhuber_15_6.0_0_1_epoch_17_metrics_0.806.param',
 ]

submitList = []
for paramFile in paramFileList:
    paramFile = paramFile[:-6]
    paramList = re.split("_|/",paramFile)
    embeddingSize = int(paramList[2])
    isSampleTag = int(paramList[4])
    isAddConstant = int(paramList[5])
    validAUC = float(paramList[9])
    pmfModel = advancedCF(embeddingSize, maxUserCount,maxTagCount, isSampleTag, isAddConstant)
    pmfModel.load_parameters(paramFile+".param", ctx=modelCtx)
    cf = CFTrainer(pmfModel,  dataCtx, modelCtx)
    submit = cf.getSubmit(test)[:,2:3]
    submitList.append(submit)

ensembleSubmit2 = np.hstack(submitList).mean(axis=1)

ensembleSubmit = 0.4*ensembleSubmit+0.6*ensembleSubmit2
ensembleSubmit[ensembleSubmit>1] = 1
ensembleSubmit[ensembleSubmit<0]=0
submit = np.hstack([test, ensembleSubmit.reshape(-1,1)])
fileName="submit"+str(datetime.today().strftime('%Y%m%d'))+".csv"
np.savetxt(fileName, submit, fmt=('%d', '%d', '%f'))


