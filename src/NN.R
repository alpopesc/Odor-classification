library(keras)
library(readr)
library(purrr)
library(reticulate)
library(embed)
source("src/TF_BugFix.R")
use_condaenv('r-tensorflow')
tensorflow::use_session_with_seed(42)
remove_zeros <- function(data){
  vec <- c()
  for(i in colnames(data)){
    if(all(data[i] == 0)){
      data[i] <- NULL
    }
  }
  data
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
  for(i in colnames(data)){
    if(sd(unlist(data[i])) == 0){
      data[i] <- NULL
    }
  }
  data
}

data <- read_csv("data/training_data.csv")
data$Intensity <- as.numeric(as.logical(data$Intensity == "high" ))
#data$SWEETORSOUR <- as.numeric(data$SWEETORSOUR)
data$VALENCE.PLEASANTNESS <- NULL
data <- na.omit(data)
data <- remove_zeros(data)
data <- na.omit(data)
data <- remove_constants(data)
View(data)

#Split data in test data and training data

idx <- sample(nrow(data), nrow(data)/4)
test <- data[idx,]
train <- data[-idx,]

Y <- scale(data$SWEETORSOUR)
data$SWEETORSOUR <- NULL

nn <- keras_model_sequential() %>%
  layer_dense(units = 30, activation = 'relu', input_shape = ncol(data)) %>%
  layer_dense(units = 30, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')
summary(nn)


nn %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(lr = .1), # stochastic gradient descent
  metrics = c('accuracy')
)


callback <- callback_early_stopping(monitor = "val_loss")

history <- nn %>% fit(
  as.matrix(data[,1:ncol(data)]),
  as.logical(Y),
  batch_size = 200, # all data points used in each iteration (regular gradient descent)
  epochs = 100,
  callbacks = callback,
  validation_split = 0.5 # use all data for training, none for validation.
)
plot(history)
View(history)



nn1 <- keras_model_sequential() %>%
  layer_dense(units = 30, activation = 'relu', input_shape = ncol(train)-1) %>%
  layer_dense(units = 30, activation = 'relu') %>%
  layer_dense(units = 1, activation = 'sigmoid')
summary(nn)


nn1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam', # stochastic gradient descent
  metrics = c('accuracy')
)

history2 <- nn1 %>% fit(
  as.matrix(train[,-2]),
  as.logical(train$SWEETORSOUR),
  batch_size = 200, # all data points used in each iteration (regular gradient descent)
  epochs = 100,
  callbacks = callback,
  validation_split = 0.1 # use all data for training, none for validation.
)
plot(history2)

nn.pred <- predict(conv.nn, test)
mean((nn.pred[,1] > 0.5) == y_test)

nn.ROCRpred <- prediction(nn.pred, as.numeric(test$SWEETORSOUR))
nn.ROCRperf <- performance(nn.ROCRpred, 'tpr', 'fpr')
plot(nn.ROCRperf, lwd = 2, col = "red")

#next step is to use polynomial transformation and proceed with regularization

