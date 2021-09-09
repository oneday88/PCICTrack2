import math
import numpy as np

import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn

class advancedCF(nn.Block):
    def __init__(self, embeddingSize, userinput_dim, iteminput_dim, isSampleTag=False, isAddConstant=False,  **kwargs):
        super(advancedCF, self).__init__(**kwargs)
        with self.name_scope():
            ### 
            self.TagNum = 5 if isSampleTag else 8
            self.isAddConstant = isAddConstant
            if(self.isAddConstant<=2):
                self.shapeSize = self.TagNum+1
            else:
                self.shapeSize = self.TagNum+2
            ### The user and item
            self.userEmbedding = nn.Embedding(input_dim=userinput_dim, output_dim=embeddingSize)
            self.itemEmbedding = nn.Embedding(input_dim=iteminput_dim, output_dim=embeddingSize)
            self.userBias = nn.Embedding(userinput_dim, 1)
            self.itemBias = nn.Embedding(iteminput_dim, 1)
            self.W = self.params.get('statsW', shape=(self.shapeSize,), init=mx.init.Uniform(0.1))

    def forward(self, x, y):
        # The pred 1
        userVecs = self.userEmbedding(x[:, 0])
        itemVecs = self.itemEmbedding(x[:, 1])
        dotProduct = nd.multiply(userVecs,itemVecs).sum(axis=1)
        bU = self.userBias(x[:, 0])[:,0]
        bI = self.itemBias(x[:, 1])[:,0]
        pred = dotProduct+bU+bI
        # The pred 2
        userVecs2 = self.userEmbedding(y[:,0])
        bU2 = self.userBias(y[:, 0])[:,0]
        # The tag part
        tagPredDt = []
        for i in range(self.TagNum):
            subItemVec = self.itemEmbedding(y[:,i+1])
            subBI = self.itemBias(y[:,i+1])[:,0]
            subPred =  nd.multiply(userVecs2,subItemVec).sum(axis=1)+bU2+subBI
            tagPredDt.append(subPred.reshape(-1,1))
        tagPredDt = nd.concat(*tagPredDt,dim=1)
        if(self.isAddConstant==1):
            tagPredDt  = nd.concat(tagPredDt, y[:,(self.TagNum+1):(self.TagNum+2)])
        elif(self.isAddConstant==2):
            tagPredDt  = nd.concat(tagPredDt, y[:,(self.TagNum+2):(self.TagNum+3)])
        else:
            tagPredDt  = nd.concat(tagPredDt, y[:,(self.TagNum+1):(self.TagNum+2)])
            tagPredDt  = nd.concat(tagPredDt, y[:,(self.TagNum+2):(self.TagNum+3)])
        pred2 = nd.sum(nd.multiply(tagPredDt, self.W.data()), axis=1)
        return pred, pred2

    def predict(self,x):
        userVecs = self.userEmbedding(x[:, 0])
        itemVecs = self.itemEmbedding(x[:, 1])
        dotProduct = nd.multiply(userVecs,itemVecs).sum(axis=1)
        bU = self.userBias(x[:, 0])[:,0]
        bI = self.itemBias(x[:, 1])[:,0]
        pred = dotProduct+bU+bI
        return pred
    
    def getEmbedding(self, x):
        userVecs = self.userEmbedding(x[:, 0])
        itemVecs = self.itemEmbedding(x[:, 1])
        dotProduct = nd.multiply(userVecs,itemVecs).sum(axis=1)
        bU = self.userBias(x[:, 0])[:,0]
        bI = self.itemBias(x[:, 1])[:,0]
        return dotProduct, bU, bI

class CausalE(nn.Block):
    def __init__(self, embeddingSize, userinput_dim, iteminput_dim, isSampleTag=False, isAddConstant=False,  **kwargs):
        super(CausalE, self).__init__(**kwargs)
        with self.name_scope():
            ### 
            self.TagNum = 5 if isSampleTag else 8
            self.isAddConstant = isAddConstant
            self.shapeSize = self.TagNum+self.isAddConstant
            ### The user and item
            self.userEmbedding = nn.Embedding(input_dim=userinput_dim, output_dim=embeddingSize)
            self.itemEmbedding = nn.Embedding(input_dim=iteminput_dim, output_dim=embeddingSize)
            self.itemEmbedding2 = nn.Embedding(input_dim=iteminput_dim, output_dim=embeddingSize)
            self.userBias = nn.Embedding(userinput_dim, 1)
            self.itemBias = nn.Embedding(iteminput_dim, 1)
            self.W = self.params.get('statsW', shape=(self.shapeSize,), init=mx.init.Uniform(0.1))

    def forward(self, x, y):
        # The pred 1
        userVecs = self.userEmbedding(x[:, 0])
        itemVecs = self.itemEmbedding(x[:, 1])
        itemVecs2 = self.itemEmbedding2(x[:, 1])
        dotProduct = nd.multiply(userVecs,itemVecs).sum(axis=1)
        dotProduct2 = nd.multiply(userVecs,itemVecs2).sum(axis=1)
        bU = self.userBias(x[:, 0])[:,0]
        bI = self.itemBias(x[:, 1])[:,0]
        pred = dotProduct+bU+bI
        pred2 = dotProduct2+bU+bI
        itemVecGap = itemVecs-itemVecs2
        # The pred 3
        userVecs2 = self.userEmbedding(y[:,0])
        bU2 = self.userBias(y[:, 0])[:,0]
        # The tag part
        tagPredDt = []
        for i in range(self.TagNum):
            subItemVec = self.itemEmbedding(y[:,i+1])
            subBI = self.itemBias(y[:,i+1])[:,0]
            subPred =  nd.multiply(userVecs2,subItemVec).sum(axis=1)+bU2+subBI
            tagPredDt.append(subPred.reshape(-1,1))
        tagPredDt = nd.concat(*tagPredDt,dim=1)
        #The constant part
        if(self.isAddConstant):
            tagPredDt  = nd.concat(tagPredDt, y[:,(self.TagNum+1):(self.TagNum+2)])
        pred3 = nd.sum(nd.multiply(tagPredDt, self.W.data()), axis=1)
        return pred, pred2, pred3,itemVecGap

    def predict(self,x):
        userVecs = self.userEmbedding(x[:, 0])
        itemVecs = self.itemEmbedding(x[:, 1])
        dotProduct = nd.multiply(userVecs,itemVecs).sum(axis=1)
        bU = self.userBias(x[:, 0])[:,0]
        bI = self.itemBias(x[:, 1])[:,0]
        pred = dotProduct+bU+bI
        return pred
