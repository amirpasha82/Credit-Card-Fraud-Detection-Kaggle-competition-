rm(list = ls())
#setwd("~/Dropbox/creditcardfraud")

library(data.table)
library(h2o)
library(DMwR)

credit = fread("creditcard.csv", header = T, sep = ",")

# Check if the data is imbalanced or not
credit$Class = as.factor(credit$Class)
prop.table(table(credit$Class))


# train/test split
split.ratio = 0.8
train.index = sample(1:nrow(credit), floor(split.ratio*nrow(credit)), replace = FALSE)
train.data = credit[train.index, ]
test.data = credit[-train.index, ]

# super-sampling via SMOTE
smote.data = SMOTE(Class ~ ., credit, perc.over = 600,perc.under=125)


# initialize h2o and establish connection
h2o.init(ip = 'localhost', port = 54321, nthreads= -1,
         max_mem_size = '8g')
# transform datframes to h2o-type data
train.data.h2o = as.h2o(train.data)
test.data.h2o = as.h2o(test.data)
smote.data.h2o = as.h2o(smote.data)
# model fitting using RF
set.seed(1234)
model_rf.h2o = h2o.randomForest(y = "Class", training_frame = smote.data.h2o, nfolds = 10, ntrees = 1000)
# model testing (on the whole test Credit data)
perf_rf.h2o = h2o.performance(model_rf.h2o, newdata = test.data.h2o)
