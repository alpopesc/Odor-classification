library(keras)
library(readr)
library(purrr)
library(dplyr)
library(reticulate)
library(embed)
library(ROCR)
library(OutlierDetection)
use_condaenv('r-tensorflow')
tensorflow::use_session_with_seed(100)

#inp <- best.lasso[["beta"]]@i

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



library(tensorflow)


K <- keras::backend()

loss <- function(y_true,y_pred){
  gamma=2.
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

View(data)
write.table(data, "train.csv", row.names=FALSE, sep=",")
write.table(final_test, "test.csv", row.names=FALSE, sep=",")


#Split data in test data and training data
s <- sample(nrow(data), nrow(data))
data <- data[s,]
idx <- sample(nrow(data), nrow(data)/4)
test <- data[idx,]
train <- data[-idx,]



callback <- callback_early_stopping(
  monitor = "val_loss",
  patience = 15,
  restore_best_weights = TRUE)


nn1 <- keras_model_sequential() %>%
  layer_dense(units = 300, activation = 'swish', bias_regularizer = regularizer_l2(0.01), kernel_regularizer = regularizer_l2(0.01), input_shape = ncol(data)-1) %>%
  layer_dropout(rate = 0.2)%>%
  layer_dense(units = 200, activation = 'swish', bias_regularizer = regularizer_l2(0.01), kernel_regularizer = regularizer_l2(0.01))%>%
  layer_dropout(rate = 0.1)%>%
  layer_dense(units = 100, activation = 'swish', bias_regularizer = regularizer_l2(0.01), kernel_regularizer = regularizer_l2(0.01))%>%
  layer_batch_normalization()%>%
  layer_dense(units = 1, activation = 'sigmoid') 

summary(nn1)

nn1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam', # stochastic gradient descent
  metrics = c('accuracy')
)

history2 <- nn1 %>% fit(
  as.matrix(data[,-1]),
  as.logical(data[,1]),
  batch_size = 50, 
  epochs = 10,
  #callbacks = callback,
  validation_split = 0.1,
  verbose = 2
)
plot(history2)



nn1.pred <- predict(nn1, as.matrix(final_test))
View(nn1.pred)
final.pred <- data.frame(cbind(nn1.pred))
final.pred$Id <- 1:68
names(final.pred)[1] <- "SWEETORSOUR"
final.pred <- final.pred[c("Id", "SWEETORSOUR")]
write.table(final.pred, "sixth_try.csv", row.names=FALSE, sep=",")
View(final.pred)
mean((nn1.pred[,1] > 0.5) == (test[,2]))


nn1.ROCRpred <- prediction(nn1.pred, as.numeric(test[,2]))
nn1.ROCRperf <- performance(nn1.ROCRpred, 'tpr', 'fpr')
plot(nn1.ROCRperf, lwd = 2, col = "red")
attr(performance(nn1.ROCRpred, 'auc'), 'y.values')





#Hyperparameter tuning
#Initial conditions -> KMeans Initialisation  or Glorot or Initialization for focal Loss
#Number of Predictors including Polynomial Transformation()
#Regularization L1/L2
#Types of layers and layer quantity
#Consider Bootstrapping

