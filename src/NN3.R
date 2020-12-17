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
tensorflow::use_session_with_seed(100)



remove_zeros <- function(data){
  vec <- c()
  for(i in colnames(data)){
    if(all(data[,i] == 0)){
      vec <- append(i,vec)
    }
  }
  vec
}


#checking all the different datatypes

data_types <- function(frame) {
  res <- lapply(frame, class)
  res_frame <- data.frame(unlist(res))
  barplot(table(res_frame), main="Data Types", col="steelblue", ylab="Number of Features")
  c(sum(res_frame == "numeric"), sum(res_frame != "numeric"))
}



library(purrr)
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

x1 <- data.frame(X = 5*rnorm(100), Y = 4*rnorm(100))
View(x1)
x1.scaled <- scale(x1)
View(x1.scaled)
get.scale(x1.scaled)
x2 <- data.frame(X = 10*rnorm(70) + .2, Y = rep(0,70))
View(x2)
x2.scaled <- scale.as(x2, x1.scaled)
View(x2.scaled)

library(tensorflow)


K <- keras::backend()



loss <- function(y_true,y_pred){
  gamma=2
  alpha=.25
  pt_1 <- tf$where(tf$equal(y_true,1),y_pred,tf$ones_like(y_pred))
  
  pt_0 <- tf$where(tf$equal(y_true,0),y_pred,tf$ones_like(y_pred))
  
  #clip to prevent NaNs and Infs
  
  epsilon <- K$epsilon()
  
  pt_1 <- K$clip(pt_1,epsilon,1.-epsilon)
  pt_0 <- K$clip(pt_0,epsilon,1.-epsilon)
  
  return(-K$mean(alpha*K$pow(1.-pt_1,gamma)*K$log(pt_1))-K$mean((1-alpha)*K$pow(pt_0,gamma)*K$log(1.-pt_0)))
  
}






#Preprocessing of data
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
data <- cbind(data[,c(1,2)],scale(data[,c(-1,-2)]))
anyNA(data)
data <-data[ , colSums(is.na(data)) == 0]
anyNA(data)
#outliers <- depthout(data)
#data <- data[-outliers[["Location of Outlier"]],] Removes outliers

library(caret)
data_cor <- cor(as.matrix(data))
hc <- findCorrelation(data_cor, cutoff=0.80) 
data <- data[,-c(sort(hc))]
View(data)

#outliers <- depthout(data)
#data <- data[-outliers[["Location of Outlier"]],] 


anyNA(data)

#downloading test_data
final_test <- read_csv(file.path("data/test_data.csv"))
final_test$VALENCE.PLEASANTNESS <- NULL
final_test$Id <- NULL
final_test$Intensity <- as.numeric(as.logical(final_test$Intensity == "high" ))
Test_zeros <- names(final_test) %in% i_zeros
final_test <- final_test[!Test_zeros]
Test_const <- names(final_test) %in% i_const
final_test <- final_test[!Test_const]


View(data)

d <- cbind(final_test[,1], scale.as(data.frame(final_test[,-1]), data.frame(data[, c(-1,-2)])))
View(d)
data$nC



jj <- remove_zeros(final_test)
JJ_t <- names(final_test) %in% jj
final_test <- final_test[!JJ_t]
kk <- remove_constants(final_test)
KK_t <- names(final_test) %in% kk
final_test <- final_test[!KK_t]

JJ <- names(data) %in% jj
data <- data[!JJ]
KK <- names(data) %in% kk
data <- data[!KK]
final_test <- scale(final_test)

write.table(data, "train.csv", row.names=FALSE, sep=",")
write.table(final_test, "test.csv", row.names=FALSE, sep=",")

View(final_test)



#Split data in test data and training data
s <- sample(nrow(data), nrow(data))
data <- data[s,]
idx <- sample(nrow(data), nrow(data)/10)
val <- data[idx,]
train <- data[-idx,]



callback <- callback_early_stopping(
  monitor = "val_loss",
  patience = 15,
  restore_best_weights = TRUE)


nn1 <- keras_model_sequential() %>%
  layer_dense(units = 27, activation = 'swish', bias_regularizer = regularizer_l2(0), kernel_regularizer = regularizer_l2(0), input_shape = ncol(data)-1) %>%
  layer_dropout(rate = 0.7)%>%
  layer_dense(units = 50, activation = 'swish', bias_regularizer = regularizer_l2(0), kernel_regularizer = regularizer_l2(0))%>%
  layer_dropout(rate = 0.1548)%>%
  layer_dense(units = 19.0750, activation = 'swish', bias_regularizer = regularizer_l2(0.4885), kernel_regularizer = regularizer_l2(0.4885))%>%
  layer_batch_normalization()%>%
  layer_dense(units = 1, activation = 'sigmoid') 


nn1 %>% compile(
  loss = loss,
  optimizer = 'adam', # stochastic gradient descent
  metrics = tf$keras$metrics$AUC()
)

history2 <- nn1 %>% fit(
  as.matrix(data[,-1]),
  as.logical(data[,1]),
  batch_size = nrow(data), 
  epochs = 300,
  #callbacks = callback,
  validation_split = 0,
  verbose = 2,
  class_weight = list("0"=0.82,"1"=1.283)
)

plot(history2)
nn1.pred <- predict(nn1, as.matrix(val[,-1]))
nn1.ROCRpred <- prediction(nn1.pred, as.logical(val[,1]))
nn1.ROCRperf <- performance(nn1.ROCRpred, 'tpr', 'fpr')
nn1.pred_t <- predict(nn1, as.matrix(train[,-1]))
nn1.ROCRpred_t <- prediction(nn1.pred_t, as.logical(train[,1]))
nn1.ROCRperf_t <- performance(nn1.ROCRpred_t, 'tpr', 'fpr')
plot(nn1.ROCRperf_t, lwd = 2, col = "blue")
attr(performance(nn1.ROCRpred_t, 'auc'), 'y.values')

plot(nn1.ROCRperf, lwd = 2, col = "red")
attr(performance(nn1.ROCRpred, 'auc'), 'y.values')


library(dplyr)
x <- colnames(data[,-1])
View(x)
final_test <- subset(final_test, select = x)
View(final_test)
View(data)


View(final_test)

nn1.pred <- predict(nn1, as.matrix(final_test))
View(nn1.pred)
final.pred <- data.frame(cbind(nn1.pred))
final.pred$Id <- 1:68
names(final.pred)[1] <- "SWEETORSOUR"
final.pred <- final.pred[c("Id", "SWEETORSOUR")]
write.table(final.pred, "eight_try.csv", row.names=FALSE, sep=",")
View(final.pred)
mean((nn1.pred[,1] > 0.5) == (test[,2]))

d1 <- as.array(data[,2])
d2 <- as.array(data[,3])
View(d)

