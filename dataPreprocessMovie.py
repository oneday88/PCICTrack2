import os, random
import _pickle as pickle

import numpy as np
import pandas as pd

"""
Load the data and merge
"""
rating = pd.read_csv('./train/rating.txt', dtype=int, sep=' ',header=None)
bigtag = pd.read_csv('./train/bigtag.txt',dtype=int, sep=' ',header=None)
choicetag = pd.read_csv('./train/choicetag.txt',dtype=int, sep = ' ', header=None)
moviedata = pd.read_csv('./train/movie.txt',dtype=int, sep = ' ', header=None)

validDt = pd.DataFrame(np.loadtxt('./valid/validation.txt',dtype=int))
testDt = pd.DataFrame(np.loadtxt('./test_phase2.txt', dtype=int))

rating.columns = ["userid", "movieid", "rating"]
bigtag.columns = ["userid", "movieid", "tagid"]
choicetag.columns = ["userid", "movieid", "tagid"]
moviedata.columns=["movieid", "tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7", "tag8"]

validDt.columns = ["userid","tagid","label"]
testDt.columns = ["userid","tagid"]

"""
The auxiliary rating data
"""
tagList = ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7","tag8"]
rating  = pd.merge(rating, moviedata, how='left', on=['movieid'])
rating['movieAveRate'] = rating['rating'].groupby(rating['movieid']).transform('mean')

featureList = ["userid","tagid","movieid","rating","movieAveRate"]
movieTagDt =[]
for tagIndex in tagList:
    subColumns = featureList.copy()
    subColumns[1] = tagIndex
    subDt = rating[subColumns]
    subDt.columns = featureList
    movieTagDt.append(subDt)
movieTagDt =  pd.concat(movieTagDt)

"""
Create the training data for matrix factorization
"""
bigtag['mark'] = 'big'
choicetag['mark'] = 'choice'
totalTag = pd.concat([bigtag, choicetag])

totalTag = pd.merge(totalTag, moviedata, how='left', on=['movieid'])
totalTag['label'] = 1
totalTag.loc[totalTag.tagid==-1,'label'] = 0
tagList = ["tag1", "tag2", "tag3", "tag4", "tag5", "tag6", "tag7","tag8"]

"""
we split the dataset into 3 parts:
    (1) big, like
    (2) -1 dislike all
    (3) choice, like, dislike
"""
featureList = ["userid", "tagid","label"]
totalTagDt1 = totalTag.loc[(totalTag.mark =='big') & (totalTag.tagid != -1), featureList]
totalTagDt2 = totalTag.loc[totalTag.tagid == -1, ]
totalTagDt3 = totalTag.loc[(totalTag.mark=='choice') & (totalTag.tagid != -1),]

#The negative -1 data
trainDt = []
for tagIndex in tagList:
    subColumns = featureList.copy()
    subColumns[1] = tagIndex
    subDt = totalTagDt2[subColumns]
    subDt.columns = featureList
    trainDt.append(subDt)
totalTagDt2 =  pd.concat(trainDt)

# The choice tag data
useridList = totalTagDt3['userid'].unique()
trainDt2 = []
for subUserid in useridList:
   subDt = totalTagDt3.loc[totalTagDt3.userid==subUserid,]
   subMovieidList = subDt['movieid'].unique()
   for subMovieid in subMovieidList:
        subMovieTagList = moviedata.loc[moviedata.movieid==subMovieid,tagList].values[0]
        positiveTagList = subDt.loc[subDt.movieid==subMovieid,'tagid'].values
        negativeTagList = subMovieTagList[~np.isin(subMovieTagList, positiveTagList)]
        if(len(positiveTagList)>0):
            subsubDt1 = pd.DataFrame({'userid': subUserid, 'tagid': positiveTagList, 'label': 1})
            trainDt2.append(subsubDt1)
        if(len(negativeTagList)):
            subsubDt2 = pd.DataFrame({'userid': subUserid, 'tagid': negativeTagList, 'label': 0})
            trainDt2.append(subsubDt2)
totalTagDt3 = pd.concat(trainDt2)

totalDt = pd.concat([totalTagDt1, totalTagDt2, totalTagDt3])
totalDt.drop_duplicates(inplace=True)

"""
The data for R model
"""
totalDt = pd.merge(totalDt, movieTagDt, on=["userid","tagid"], how='inner')
validDt = pd.merge(validDt, movieTagDt, on=["userid","tagid"], how='inner')
testDt = pd.merge(testDt, movieTagDt, on=["userid","tagid"], how='inner')

paramFile='Params/Bothhuber_15_6.0_0_1_epoch_16_metrics_0.806.param'

from CFModels import advancedCF
from CFTrainer import CFTrainer
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn
embeddingSize=15
maxUserCount = 1000
maxTagCount = 1720
import mxnet as mx
modelCtx = mx.cpu()
pmfModel = advancedCF(embeddingSize, maxUserCount,maxTagCount, 0,1)
pmfModel.load_parameters(paramFile, ctx=modelCtx)

def getRModelDt(totalDt):
    totalX = nd.array(totalDt[['userid','tagid']].values)
    totalDotProduct, totalUserBias, totalItemBias = pmfModel.getEmbedding(totalX)
    totalDt['dotProduct'] = totalDotProduct.asnumpy()
    totalDt["userBias"] = totalUserBias.asnumpy()
    totalDt['itemBias'] = totalItemBias.asnumpy()
    return totalDt

totalDt = getRModelDt(totalDt)
validDt = getRModelDt(validDt)
testDt = getRModelDt(testDt)

totalDt.to_csv("totalDt.csv", sep=',', index=False)
validDt.to_csv("validDt.csv", sep=',', index=False)
testDt.to_csv("testDt.csv", sep=',', index=False)
