library(readr)
library(tidymodels)
library(ROCR)
library(glmnet)
library(caret)
library(PerformanceAnalytics)
library(tensorflow) #use_session_with_seed

use_session_with_seed(199)

#Load data

test_data <- read_csv(file.path("data/test_data.csv"))
training_data <- read_csv(file.path("data/training_data.csv"))
data <- na.omit(training_data)
data$VALENCE.PLEASANTNESS <- NULL
data$Intensity <- as.numeric(as.logical(data$Intensity == "high"))
data$SWEETORSOUR <- as.numeric(data$SWEETORSOUR)
test_data$Intensity <- as.numeric(as.logical(test_data$Intensity == "high"))

#FEATURE SELECTION BY REMOVING PREDICTORS WITH LOW VARIANCES

for(i in colnames(data[,c(-1,-2)])) {
  if(var(data[[i]]) <= 1e-5) {
     data[[i]] <- NULL
  }
}

#FEATURE SELECTION BY REMOVING CORRELATED PREDICTORS

data_cor <- cor(as.matrix(data[,3:ncol(data)]))
hc <- findCorrelation(data_cor, cutoff=0.99)
data_intensity <- data$Intensity
data_SWEETORSOUR <- data$SWEETORSOUR
data <- data[,-c(sort(hc))]
data$SWEETORSOUR <- data_SWEETORSOUR
data$Intensity <- data_intensity

#FEATURE ENGINEERING BY CHECKING SKEWNESS & APPLY LOG

get_skewed_names <- function(d){
  ret <- c()
  ret <- NULL
  for (i in colnames(d[,c(-1,-2)])) {
    if (abs(skewness(d[[i]], type = 2)) > 0.7) {
      ret <- append(ret, i)
    }
  }
  ret
}

return_logd_data <- function(d, skw_names){
  for(i in skw_names){
    if (0 %in% d[[i]]) {
      for (j in 1:nrow(d)) {
        if (d[j,i] != 0) {
          d[j,i] <- log(d[j,i])
        }
      }
    } else {
      d[[i]] <- log(d[[i]])
    }
  }
  return(d)
}

sk_names <- get_skewed_names(data)
data <- return_logd_data(data, sk_names)

##########SIMPLE REGRESSION

# cv_data <- vfold_cv(data, v = 10) # create the 10 folds
# AUC_values <- vector()
# ntrees <- vector()
# for (i in 1:10) {
#   training_set <- analysis(cv_data$splits[[i]])
#   validation_set <- assessment(cv_data$splits[[i]])
#   
#   logreg.fit <- glm(SWEETORSOUR ~ ., training_set, family = 'binomial')
#   logreg.pred <- predict(logreg.fit, validation_set, type = "response")
#   ROCR.pred.logreg <- prediction(logreg.pred, validation_set$SWEETORSOUR)
#   AUC_values <- c(AUC_values, performance(ROCR.pred.logreg, 'auc')@y.values[[1]])
# }
# mean(AUC_values)
# 
# logreg.fit <- glm(SWEETORSOUR ~ ., data.train, family = 'binomial')
# logreg.pred <- predict(logreg.fit, data.validation, type = "response")
# ROCR.pred.logreg <- prediction(logreg.pred, data.validation$SWEETORSOUR)
# print(paste('LogReg:', performance(ROCR.pred.logreg, 'auc')@y.values, sep = ' '))

##########SIMPLE REGRESSION

# Splitting the data into training and validation sets

idx.train <- sample(nrow(data), nrow(data)*0.75)
data.train <- data[idx.train,]
data.validation <- data[-idx.train,]

x <- data.matrix(data.train[,names(data.train) != "SWEETORSOUR"])
x[!is.finite(x)] <- 0
y <- data.train$SWEETORSOUR
x.validation <- data.matrix(data.validation[,names(data.validation) != "SWEETORSOUR"])
x.validation[!is.finite(x.validation)] <- 0
y.validation <- data.validation$SWEETORSOUR

#LOGISTIC REGRESSION WITH LASSO REGULARIZATION

cv.lasso <- cv.glmnet(x, y , alpha = 1, nfold = 10)
plot(cv.lasso)
best.lasso <- glmnet(x, y, alpha = 1, lambda = cv.lasso$lambda.min)
lasso.pred <- predict(best.lasso, s = cv.lasso$lambda.min, newx = x.validation, type = "response")
plot(performance(prediction(lasso.pred, y.validation), 'tpr', 'fpr'))
auc.lasso <- performance(prediction(lasso.pred, y.validation), measure = 'auc')
auc.lasso.value <- auc.lasso@y.values[[1]]

print(paste('LogReg + Lasso AUC :', auc.lasso.value, sep = ' '))

