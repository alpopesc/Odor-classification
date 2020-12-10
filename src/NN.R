library(keras)
library(readr)
library(purrr)
library(reticulate)
library(embed)
library(ROCR)
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



#Preprocessing of data
data <- read_csv("data/training_data.csv")
data$Intensity <- as.numeric(as.logical(data$Intensity == "high" ))
data$SWEETORSOUR <- as.numeric(data$SWEETORSOUR)
data$VALENCE.PLEASANTNESS <- NULL
data <- na.omit(data)
data <- remove_zeros(data)
data <- remove_constants(data)
data <- cbind(data[,c(1,2)],scale(data[,c(-1,-2)]))
data <-data[ , colSums(is.na(data)) == 0]
anyNA(data)



#Split data in test data and training data
s <- sample(nrow(data), nrow(data))
data <- data[s,]
idx <- sample(nrow(data), nrow(data)/4)
test <- data[idx,]
train <- data[-idx,]


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


q <- scale(train[,-2])
q <-q[ , colSums(is.na(q)) == 0]
anyNA(q)
ncol(q)




nn1 <- keras_model_sequential() %>%
  layer_dense(units = 100, kernel_regularizer = regularizer_l2(l = 0.03), activation = 'swish', input_shape = ncol(train)-1) %>%
  layer_dropout(rate = 0.01) %>%
  layer_dense(units = 100, kernel_regularizer = regularizer_l2(l = 0.03), activation = 'swish') %>%
  layer_dropout(rate = 0.01) %>%
  layer_dense(units = 100, kernel_regularizer = regularizer_l2(l = 0.03), activation = 'swish') %>%
  layer_dropout(rate = 0.01) %>%
  layer_dense(units = 100, kernel_regularizer = regularizer_l2(l = 0.03), activation = 'swish') %>%
  layer_dropout(rate = 0.01) %>%
  layer_dense(units = 100, kernel_regularizer = regularizer_l2(l = 0.02), activation = 'swish') %>%
  layer_dropout(rate = 0.01) %>%
  layer_dense(units = 1, activation = 'sigmoid') 

summary(nn1)



nn1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam', # stochastic gradient descent
  metrics = c('accuracy')
)



history2 <- nn1 %>% fit(
  as.matrix(train[,-2]),
  as.logical(train[,2]),
  batch_size = nrow(train), 
  epochs = 200,
  #callbacks = callback,
  validation_split = 0 
)
plot(history2)

nn1.pred <- predict(nn1, as.matrix(test[,-2]))
mean((nn1.pred[,1] > 0.5) == (test[,2]))

f <- rep(1,nrow(nn1.pred))-nn1.pred

nn1.ROCRpred <- prediction(nn1.pred, as.numeric(test[,2]))
nn1.ROCRperf <- performance(nn1.ROCRpred, 'tpr', 'fpr')
plot(nn1.ROCRperf, lwd = 2, col = "red")
attr(performance(nn1.ROCRpred, 'auc'), 'y.values')


#next step is to use polynomial transformation and proceed with regularization


get_Nodes <- function(layers, X, Y, N, rng){
  matrix(sample.int(rng, N*layers, TRUE), N, layers)
  for(i in 1:N){
    
  }
}

evaluate_nn <- function(X,Y,nodes,lambda){
  if(length(nodes) = 2){
    nn1 <- keras_model_sequential() %>%
      layer_dense(units = nodes[1], kernel_regularizer = regularizer_l2(l = lambda[1]), activation = 'relu', input_shape = ncol(X)) %>%
      layer_dense(units = nodes[2] , kernel_regularizer = regularizer_l2(l = lambda[2]), activation = 'relu') %>%
      layer_dense(units = 1, activation = 'sigmoid') 
    
    nn1 %>% compile(
      loss = 'binary_crossentropy',
      optimizer = 'adam', # stochastic gradient descent
      metrics = c('accuracy')
    )
    
    history2 <- nn1 %>% fit(
      as.matrix(train[,-2]),
      as.logical(train$SWEETORSOUR),
      batch_size = 100, 
      epochs = 10,
      #callbacks = callback,
      validation_split = 0.2 
    )
  }
}

d <- get_Nodes(3,1,1,10,100)
d[1,][1]



#Hyperparameter tuning
#Initial conditions -> KMeans Initialisation or Glorot
#Number of Predictors including Polynomial Transformation()
#Regularization L1/L2
#Types of layers and layer quantity
#Consider Bootstrapping

