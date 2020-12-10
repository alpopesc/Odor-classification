library(readr)
library(dplyr)
library(tidymodels)
library(leaps)
library (evaluate)
library(ROCR)
library(glmnet)
library(MASS)

#TO DO
#Vector features (week 5)
#Gradient descent (regularization, week 6)

#Load data

test_data <- read_csv(file.path("data/test_data.csv"))
data <- read_csv(file.path("data/training_data.csv"))
data <- na.omit(data)

#Remove zeros columns
remove_zeros_col <- function(data) {
  for (i in colnames(data)) {
    if (class(data[[i]]) == "numeric" & all(data[[i]] == 0)) {
      data[[i]] <- NULL
    }
  }
  data
}

data <- remove_zeros_col(raw_data)
data$VALENCE.PLEASANTNESS <- NULL

# Splitting the data into training and validation sets

set.seed(199) 
#seed(199) --> df = 49/63 et AUC 0.608
#LOOCV (nfold = nrow(x)) -> df = 63 AUC = 0.606
#seed(12) --> df = 6 et AUC 0.52
#seed(123) --> df = 50 et AUC 0.587
idx.train <- sample(nrow(data), nrow(data)*0.75)
data.train <- data[idx.train,]
data.validation <- data[-idx.train,]

#Lasso regression
x <- data.matrix(data.train[,names(data.train) != "SWEETORSOUR"])
y <- data.train$SWEETORSOUR
cv.lasso <- cv.glmnet(x, y , alpha = 1, nfold = 10)
plot(cv.lasso)
best.lasso <- glmnet(x, y, alpha = 1, lambda = cv.lasso$lambda.min)
#outputs number of df
best.lasso$df
lasso.pred <- predict(best.lasso, s = cv.lasso$lambda.min, newx = data.matrix(data.validation[,names(data.validation) != "SWEETORSOUR"]), type = "response")
plot(performance(prediction(lasso.pred, data.validation$SWEETORSOUR), 'tpr', 'fpr'))
auc.lasso <- performance(prediction(lasso.pred, data.validation$SWEETORSOUR), measure = 'auc')
#AUC
auc.lasso.value <- auc.lasso@y.values[[1]] 

#Let's check if training's AUC is indeed higher than validation's AUC

lasso.pred.train <- predict(best.lasso, s = cv.lasso$lambda.min, newx = x, type = "response")
auc.lasso.training <- performance(prediction(lasso.pred.train, y), measure = 'auc')
auc.lasso.value.training <- auc.lasso.training@y.values[[1]] 
# AUC = 0.77 with seed(199)

#Let's fit our model on all the training set to make predictions on test_data

bestest.lasso <- glmnet(data.matrix(data[,names(data) != "SWEETORSOUR"]), data$SWEETORSOUR, alpha = 1, lambda = cv.lasso$lambda.min)
#We only keep the predictors used in training_data
for (i in colnames(test_data)) {
  if (!i %in% colnames(data)) {
    test_data[[i]] <- NULL
  }
}
lasso.pred.test <- predict(bestest.lasso, s = cv.lasso$lambda.min, newx = data.matrix(test_data), type = "response")
final.pred <- data.frame(cbind(lasso.pred.test))
final.pred$Id <- 1:68
names(final.pred)[1] <- "SWEETORSOUR"
final.pred <- final.pred[c("Id", "SWEETORSOUR")]
write.table(final.pred, "first_try.csv", row.names=FALSE, sep=",")

#Important predictors name and value of beta

coeff.lasso.values <- vector()
coeff.lasso.names <- vector()
for (i in 1:nrow(coef(best.lasso))) {
  if (coef(best.lasso)[i] != 0) {
    coeff.lasso.values <- c(coeff.lasso.values, coef(best.lasso)[i])
    coeff.lasso.names <- c(coeff.lasso.names, rownames(coef(best.lasso))[i])
  }
}

#Simple logistic regression 

logreg.fit <- glm(SWEETORSOUR ~ ., data = data.train, family = 'binomial')
logreg.pred <- predict(logreg.fit, data.test, type = "response")
ROCR.pred.logreg <- prediction(logreg.pred, data.test$SWEETORSOUR)
print(paste('LogReg:', performance(ROCR.pred.logreg, 'auc')@y.values, sep = ' '))

#AUC = 0.52

#############################BROKEN CODE#####################################

#LDA 
#Cannot run it unless I remove the collinear predictors..
lda.fit <- lda(formula = SWEETORSOUR ~ Intensity + VALENCE.PLEASANTNESS + nR07 + D.Dtr07 + BIC2 + MATS1s + RDF020m + RDF055s + Mor32s + Ds + nR.Cp + nRCO + C.015 + CATS2D_07_NL + B02.N.O. + F02.N.O. + Infective.50, data = data.train, CV = TRUE)
lda.pred <- predict(lda.fit, data.test)
auc.lda <- performance(prediction(lda.pred, data.test$SWEETORSOUR), measure = 'auc')
auc.lda.value <- auc.lda@y.values[[1]]


#Model selection of degree with AIC, dunno how to do poly(multiple predictors, d)
new_data = data.frame(Y = data$SWEETORSOUR, X = subset(data, select = -SWEETORSOUR))

fit.and.aic <- function(d, data) {
  fit <- glm(data$SWEETORSOUR ~ poly(data$Sv, d, raw = TRUE), data, family = "gaussian")
  summary(fit)$aic
}
aics <- sapply(1:10, fit.and.aic, data)
plot(aics, type = "l", xlab = "degree", ylab = "AIC")
d <- which.min(aics)
points(d, aics[d])