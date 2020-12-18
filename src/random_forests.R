library(tree)
library(readr)
library(ROCR)
library(randomForest)
library(tidymodels)
library(caret)
library(PerformanceAnalytics)
library(tensorflow)

use_session_with_seed(101)

#Load data

test_data <- read_csv(file.path("data/test_data.csv"))
test_data$Intensity <- as.numeric(as.logical(test_data$Intensity == "high"))

training_data <- read_csv(file.path("data/training_data.csv"))
data <- na.omit(training_data)
data$VALENCE.PLEASANTNESS <- NULL
data$Intensity <- as.numeric(as.logical(data$Intensity == "high"))
data$SWEETORSOUR <- as.numeric(data$SWEETORSOUR)


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

#k-fold CV to evaluate model with the hyper-parameters found from RF_tuning.R
cv_data <- vfold_cv(data, v = 10) # create the 10 folds
AUC_values <- vector()
ntrees <- vector()
for (i in 1:10) {
  training_set <- analysis(cv_data$splits[[i]])
  validation_set <- assessment(cv_data$splits[[i]])
  
  rf <- randomForest(SWEETORSOUR ~ ., training_set, ntree = 125, mtry = 1, maxnodes = 14)
  predValid <- predict(rf, validation_set, type = "class")
  forests.ROCRpred <- prediction(predValid, validation_set$SWEETORSOUR)
  forests.ROCRperf <- performance(forests.ROCRpred, 'tpr', 'fpr')
  print(paste('AUC of random forests:', performance(forests.ROCRpred, 'auc')@y.values, sep = ' '))
  AUC_values <- c(AUC_values, performance(forests.ROCRpred, 'auc')@y.values[[1]])
}
mean(AUC_values)