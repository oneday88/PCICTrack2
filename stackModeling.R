library(caret)
library(reshape2)
library(lubridate)
library(data.table)
############################################################
### load the dataset
############################################################
trainStack  <- fread("totalDt.csv")
validStack  <- fread("validDt.csv")
testStack   <- fread("testDt.csv")

############################################################
### modeling with RF
############################################################
subFeatures  <- c("dotProduct","userBias", "itemBias","rating","movieAveRate")

library(lightgbm)
lgb.grid = list(
                objective = "cross_entropy",
                learning_rate = 0.01,
                boosting_type = 'gbdt',
                metric = "auc",
                feature_fraction = 0.7,
                bagging_fraction = 0.7)

dtrain   <-  lgb.Dataset(data=as.matrix(subset(trainStack, select=subFeatures)), label=trainStack$label)
dvalid   <-  lgb.Dataset(data=as.matrix(subset(validStack, select=subFeatures)), label=validStack$label)

#baseLGB     <-  lightgbm(lgb.train, nrounds = 500, params=lgb.grid)
ligMmodel   <- lgb.train(dtrain, nrounds = 760, params=lgb.grid, valids = list(train = dtrain, valid = dvalid))

validPred   <- predict(ligMmodel, as.matrix(subset(validStack, select=subFeatures)))
testPred    <- predict(ligMmodel, as.matrix(subset(testStack, select=subFeatures)))

validStack[,pred:=validPred]
testStack[, pred:=testPred]

validResult <- validStack[,list(meanPred=mean(pred), medianPred=median(pred)), by=c("userid", "tagid","dotProduct","userBias","itemBias","label")]
testResult  <- testStack[,list(meanPred=mean(pred), medianPred=median(pred)), by=c("userid", "tagid","dotProduct","userBias","itemBias")]

library(pROC)
auc(validResult$label, validResult$meanPred)
auc(validResult$label, validResult$medianPred)

bestResult  <- fread("submit20210826.csv")
setnames(bestResult, c("userid","tagid","bestPred"))

bestResult  <- testResult[bestResult, on=c("userid", "tagid")]

bestResult[, bestPred2:=bestPred]

bestResult[!is.na(dotProduct), bestPred:=0.5*bestPred+0.5*medianPred]

fwrite(subset(bestResult, select=c("userid", "tagid","bestPred")), "revision20210825.csv", sep=' ',col.names=F)

