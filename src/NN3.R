library(keras)
library(readr)
library(purrr)
library(dplyr)
library(reticulate)
library(embed)
library(ROCR)
library(OutlierDetection)
library(tensorflow)
use_condaenv('r-tensorflow')
library(caret)
tensorflow::tf$random$set_seed(55)

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
##############################################################################################
#Design of the model


callback <- callback_early_stopping(
  monitor = "val_loss",
  patience = 15,
  restore_best_weights = TRUE)

nn1 <- keras_model_sequential() %>%
  layer_dense(units = 36.681, activation = 'swish', bias_regularizer = regularizer_l2(0.3624), kernel_regularizer = regularizer_l2(0.3624), input_shape = ncol(data)-1) %>%
  layer_dropout(rate = 0.393)%>%
  layer_dense(units = 22.859, activation = 'swish', bias_regularizer = regularizer_l2(0.2483), kernel_regularizer = regularizer_l2(0.2483))%>%
  layer_dropout(rate = 0.494)%>%
  layer_dense(units = 38.5014, activation = 'swish', bias_regularizer = regularizer_l2(0), kernel_regularizer = regularizer_l2(0))%>%
  layer_batch_normalization()%>%
  layer_dense(units = 1, activation = 'sigmoid') 


nn1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam', # stochastic gradient descent
  metrics = tf$keras$metrics$AUC()
)

history2 <- nn1 %>% fit(
  as.matrix(data[,-1]),
  as.logical(data[,1]),
  batch_size = nrow(train), 
  epochs = 56,
  #callbacks = callback,
  validation_split = 0,
  verbose = 2,
  class_weight = list("0"=0.82,"1"=1.283)
)

#plot training history and training AUC
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



nn1.pred <- predict(nn1, as.matrix(final_test))
final.pred <- data.frame(cbind(nn1.pred))
final.pred$Id <- 1:68
names(final.pred)[1] <- "SWEETORSOUR"
final.pred <- final.pred[c("Id", "SWEETORSOUR")]
write.table(final.pred, "NN_postsubmission.csv", row.names=FALSE, sep=",")




