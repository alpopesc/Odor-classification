library(tree)
library(readr)
library(ROCR)
library(randomForest)
library(tidymodels)
library(caret)
library(PerformanceAnalytics)
library(tensorflow)
library(e1071)

use_session_with_seed(101)

#Load data

test_data <- read_csv(file.path("data/test_data.csv"))
training_data <- read_csv(file.path("data/training_data.csv"))
data <- na.omit(training_data)
data$VALENCE.PLEASANTNESS <- NULL

test_data$Intensity <- as.factor(as.logical(test_data$Intensity == "high"))

data$SWEETORSOUR <- make.names(data$SWEETORSOUR)
#FEATURE SELECTION BY REMOVING PREDICTORS WITH LOW VARIANCES

for(i in colnames(data[,c(-1,-2)])) {
  if(var(data[[i]]) <= 1e-5) {
    data[[i]] <- NULL
  }
}

#FEATURE SELECTION BY REMOVING CORRELATED PREDICTORS

data_cor <- cor(as.matrix(data[,c(-1,-2)]))
hc <- findCorrelation(data_cor, cutoff=0.99) 
data_Intensity <- data$Intensity
data_SWEETORSOUR <- data$SWEETORSOUR
data <- data[,-c(sort(hc))]
data$Intensity <- data_Intensity
data$SWEETORSOUR <- data_SWEETORSOUR

#FEATURE ENGINEERING BY SCALING THE DATA

scale(data)

# Splitting the data into training and validation sets

idx.train <- sample(nrow(data), nrow(data)*0.75)
data.train <- data[idx.train,]
data.validation <- data[-idx.train,]
x.train <- data.frame(data.train[,colnames(data.train) != "SWEETORSOUR"])

#TUNING HYPER-PARAMETERS
trControl <- trainControl(method = "cv",
                          number = 10,
                          search = "grid",
                          summaryFunction = twoClassSummary,
                          classProbs = TRUE)
require(caret) 

#Find best mtry

tuneGrid <- expand.grid(.mtry = c(1: 10))
rf_mtry <- caret::train(x.train,
                 data.train$SWEETORSOUR,
                 method = "rf",
                 metric = "ROC",
                 tuneGrid = tuneGrid,
                 trControl = trControl,
                 importance = TRUE,
                 ntree = 300)

best_mtry <- rf_mtry$bestTune$mtry 
#best with mtry = 1

#Find best maxnodes

store_maxnode <- list()
tuneGrid <- expand.grid(.mtry = best_mtry)
for (maxnodes in c(5: 15)) {
  rf_maxnode <- caret::train(x.train,
                      data.train$SWEETORSOUR,
                      method = "rf",
                      metric = "ROC",
                      tuneGrid = tuneGrid,
                      trControl = trControl,
                      importance = TRUE,
                      maxnodes = maxnodes,
                      ntree = 300)
  current_iteration <- toString(maxnodes)
  store_maxnode[[current_iteration]] <- rf_maxnode
}
results_mtry <- resamples(store_maxnode)
summary(results_mtry)
#best with 7 maxnodes

#Find best ntree

store_maxtrees <- list()
for (ntree in c(75,100,125,150,175,200,225,250,275)) {
  set.seed(5678)
  rf_maxtrees <- caret::train(x.train,
                       data.train$SWEETORSOUR,
                       method = "rf",
                       metric = "ROC",
                       tuneGrid = tuneGrid,
                       trControl = trControl,
                       importance = TRUE,
                       maxnodes = 7,
                       ntree = ntree)
  key <- toString(ntree)
  store_maxtrees[[key]] <- rf_maxtrees
}
results_tree <- resamples(store_maxtrees)
summary(results_tree)

#best ntree = 125