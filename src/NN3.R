library(keras)
library(readr)
library(purrr)
library(dplyr)
library(reticulate)
library(embed)
library(ROCR)
library(OutlierDetection)
use_condaenv('r-tensorflow')
library(caret)
tensorflow::tf$random$set_seed(55)

get_skewed_names <- function(d){
  ret <- c()
  ret <- NULL
  for (i in colnames(d[,c(-1,-2)])) {
    if (abs(skewness(data[[i]], type = 2)) > 1) {
      ret <- append(ret, i)
    }
  }
  ret
}


return_logd_data <- function(d, skw_names){
  for(i in skw_names){
    if (0 %in% data[[i]]) {
      for (j in 1:nrow(data)) {
        if (d[j,i] != 0) {
          d[j,i] <- log(data[j,i])
        }
      }
    } else {
      d[[i]] <- log(d[[i]])
    }
  }
  return(d)
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


#Preprocessing of training data
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


#Feature engineering log
#sk_names <- get_skewed_names(data)
#data <- return_logd_data(data, sk_names)


#Scaling of data
data <- cbind(data[,2],scale(data[,-2]))
anyNA(data)



#Removing correlated predictors
data_cor <- cor(as.matrix(data[,-2]))
hc <- findCorrelation(data_cor, cutoff=0.96) +1
data <- data[,-c(sort(hc))]


#Removing outliers (Does not improve auc)
#outliers <- depthout(data)
#data <- data[-outliers[["Location of Outlier"]],] 



#Preprocessing of test_data
final_test <- read_csv(file.path("data/test_data.csv"))
final_test$VALENCE.PLEASANTNESS <- NULL
final_test$Id <- NULL
final_test$Intensity <- as.numeric(as.logical(final_test$Intensity == "high" ))
x <- colnames(data[,-2])
final_test <- subset(final_test, select = x)

#Scale Feature on test_data
final_test <- cbind(final_test[,1], scale.as(data.frame(final_test[,-1]), data.frame(data[, c(-1,-2)])))

View(data)
View(final_test)

#write.table(data, "train.csv", row.names=FALSE, sep=",")
#write.table(final_test, "test.csv", row.names=FALSE, sep=",")


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
  layer_dense(units = 36.681, activation = 'swish', bias_regularizer = regularizer_l2(0.3624), kernel_regularizer = regularizer_l2(0.3624), input_shape = ncol(train)-1) %>%
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
  as.matrix(train[,-2]),
  as.logical(train[,2]),
  batch_size = nrow(train), 
  epochs = 56,
  #callbacks = callback,
  validation_split = 0,
  verbose = 2,
  class_weight = list("0"=0.82,"1"=1.283)
)

#plot training history and training AUC
plot(history2)
nn1.pred <- predict(nn1, as.matrix(val[,-2]))
nn1.ROCRpred <- prediction(nn1.pred, as.logical(val[,2]))
nn1.ROCRperf <- performance(nn1.ROCRpred, 'tpr', 'fpr')
nn1.pred_t <- predict(nn1, as.matrix(train[,-2]))
nn1.ROCRpred_t <- prediction(nn1.pred_t, as.logical(train[,2]))
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
write.table(final.pred, "FinalNN_try.csv", row.names=FALSE, sep=",")
mean((nn1.pred[,1] > 0.5) == (test[,2]))



