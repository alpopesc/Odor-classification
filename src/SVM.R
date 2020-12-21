library(keras)
library(readr)
library(purrr)
library(dplyr)
library(reticulate)
library(embed)
library(ROCR)
library(OutlierDetection)
use_condaenv('r-tensorflow')
library(tree)
library(randomForest)
library(caret)
library(e1071)

tensorflow::tf$random$set_seed(56)
set.seed(56)



############################################## USEFUL FUNCTIONS

remove_zeros <- function(data){
  vec <- c()
  for(i in colnames(data)){
    if(all(data[,i] == 0)){
      vec <- append(i,vec)
    }
  }
  vec
}

remove_zeros <- function(data){
  vec <- c()
  for(i in colnames(data)){
    if(all(data[,i] == 0)){
      vec <- append(i,vec)
    }
  }
  vec
}

data_types <- function(frame) {
  res <- lapply(frame, class)
  res_frame <- data.frame(unlist(res))
  barplot(table(res_frame), main="Data Types", col="steelblue", ylab="Number of Features")
  c(sum(res_frame == "numeric"), sum(res_frame != "numeric"))
}

remove_constants <-function(data){
  vec <- c()
  for(i in colnames(data)){
    if(sd(unlist(data[i])) == 0){
      vec <- append(i,vec)
    }
  }
  vec
}

get.scale <- function(scaled) {
  if ("scaled:center" %in% names(attributes(scaled))) {
    center <- attr(scaled, "scaled:center")
  } else {
    center <- rep(0, ncol(scaled))
  }
  if ("scaled:scale" %in% names(attributes(scaled))) {
    list(center, attr(scaled, "scaled:scale"))
  } else {
    list(center, rep(1., length(center)))
  }
}

scale.as <- function(x, scaled) {
  s <- get.scale(scaled)
  centered <- sweep(x, 2, s[[1]])
  sweep(centered, 2, s[[2]], FUN = "/")
}


##################################################################################################

################################################## PREPORCESSING OF DATA
data <- read_csv("data/training_data.csv")
data$Intensity <- as.numeric(as.logical(data$Intensity == "high" ))
data$SWEETORSOUR <- as.numeric(data$SWEETORSOUR)
data$VALENCE.PLEASANTNESS <- NULL
data <- na.omit(data)
i_zeros <- remove_zeros(data)
I_zeros <- names(data) %in% i_zeros
data <- data[!I_zeros]
i_const <- remove_constants(data)
I_const <- names(data) %in% i_const
data <- data[!I_const]

#Removing correlated predictors
data_cor <- cor(as.matrix(data[,-2])) #not optimal but I know that Intensity is not going to be remvd
hc <- findCorrelation(data_cor, cutoff=0.99) +1
data <- data[,-c(sort(hc))]
View(data)


#Removing outliers
#outliers <- depthout(data)
#data <- data[-outliers[["Location of Outlier"]],]


#scaling the data
d_tmp <- scale(data[,-2], center = TRUE, scale = TRUE)
data <- cbind(data[,2], scale(data[,-2]))


#Preprocessing of test_data
final_test <- read_csv(file.path("data/test_data.csv"))
final_test$VALENCE.PLEASANTNESS <- NULL
final_test$Id <- NULL
final_test$Intensity <- as.numeric(as.logical(final_test$Intensity == "high" ))
x <- colnames(data[,-1])
final_test <- subset(final_test, select = x)



#Scale test data
final_test <- scale.as(final_test, d_tmp)
View(final_test)


#Split data in validation data and training data
s <- sample(nrow(data), nrow(data))
data <- data[s,]
idx <- sample(nrow(data), nrow(data)/10)
val <- data[idx,]
train <- data[-idx,]


#Note sigma = 1/gamma^2  => gamma = sqrt(1/sigma)
svmfit1 <- svm(SWEETORSOUR ~ ., data = data, kernel = "radial", cost = 0.115, rbf_sigma = 0.00167, scale = FALSE, class_weight = list("0"=0.82,"1"=1.283))
svmfit2 <- svm(SWEETORSOUR ~ ., data = data, kernel = "linear", cost = 10, scale = FALSE, class_weight = list("0"=0.82,"1"=1.283))

svm1.pred <- predict(svmfit1, as.matrix(val[,-1]))
svm1.ROCRpred <- prediction(svm1.pred, as.logical(val[,1]))
svm1.ROCRperf <- performance(svm1.ROCRpred, 'tpr', 'fpr')
svm1.pred_t <- predict(svmfit1, as.matrix(train[,-1]))
svm1.ROCRpred_t <- prediction(svm1.pred_t, as.logical(train[,1]))
svm1.ROCRperf_t <- performance(svm1.ROCRpred_t, 'tpr', 'fpr')
plot(svm1.ROCRperf_t, lwd = 2, col = "blue")
attr(performance(svm1.ROCRpred_t, 'auc'), 'y.values')
plot(svm1.ROCRperf, lwd = 2, col = "red")
attr(performance(svm1.ROCRpred, 'auc'), 'y.values')

#Use the model on the final test data
svm1.pred <- predict(svmfit1, as.matrix(final_test))
View(as.matrix(final_test))
final.pred <- data.frame(cbind(svm1.pred))
final.pred$Id <- 1:68
names(final.pred)[1] <- "SWEETORSOUR"
final.pred <- final.pred[c("Id", "SWEETORSOUR")]
write.table(final.pred, "SVM2.csv", row.names=FALSE, sep=",")
View(final.pred)
View(data)
View(final_test)



