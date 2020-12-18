library(readr)
library(tidymodels)
library(ROCR)
library(glmnet)
library(caret)
library(PerformanceAnalytics)

#Load data

test_data <- read_csv(file.path("data/test_data.csv"))
training_data <- read_csv(file.path("data/training_data.csv"))
data <- na.omit(training_data)
data$VALENCE.PLEASANTNESS <- NULL

#FEATURE SELECTION BY REMOVING PREDICTORS WITH LOW VARIANCES

for(i in colnames(data[,c(-1,-2)])) {
  if(var(data[[i]]) <= 1e-5) {
    data[[i]] <- NULL
  }
}

#FEATURE ENGINEERING BY CHECKING SKEWNESS & APPLY LOG
skewness_values <- vector()
r <- 0.50
skewness_values <- c(skewness_values, r)

while (r < 0.70) {
  r <- r + 0.02
  skewness_values <- c(skewness_values, r)
}

AUC_values <- vector()

for (i in 1:length(skewness_values)) {
  log_data <- data
  for (l in 3:ncol(log_data)) {
    if (abs(skewness(log_data[[l]])) > skewness_values[i]) {
      if (0 %in% log_data[[l]]) {
        for (j in 1:nrow(log_data)) {
          if (log_data[j,l] != 0) {
            log_data[j,l] <- log(log_data[j,l])
          }
        }
      } else {
        log_data[[l]] <- log(log_data[[l]])
      }
    }
  }
  set.seed(199) 
  idx.train <- sample(nrow(log_data), nrow(log_data)*0.75)
  data.train <- log_data[idx.train,]
  data.validation <- log_data[-idx.train,]
  
  #LOGISTIC REGRESSION WITH LASSO REGULARIZATION
  x <- data.matrix(data.train[,names(data.train) != "SWEETORSOUR"])
  x[!is.finite(x)] <- 0
  y <- data.train$SWEETORSOUR
  x.validation <- data.matrix(data.validation[,names(data.validation) != "SWEETORSOUR"])
  x.validation[!is.finite(x.validation)] <- 0
  y.validation <- data.validation$SWEETORSOUR
  
  cv.lasso <- cv.glmnet(x, y , alpha = 1, nfold = 10)
  best.lasso <- glmnet(x, y, alpha = 1, lambda = cv.lasso$lambda.min)
  lasso.pred <- predict(best.lasso, s = cv.lasso$lambda.min, newx = x.validation, type = "response")
  auc.lasso <- performance(prediction(lasso.pred, y.validation), measure = 'auc')
  auc.lasso.value <- auc.lasso@y.values[[1]] 
  AUC_values <- c(AUC_values, auc.lasso.value)
}
which.max(AUC_values)
#MAX AUC with skewness --> 0.7