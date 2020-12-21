library(keras)
library(readr)
library(purrr)
library(dplyr)
library(reticulate)
library(cloudml)
library(tfruns)
library(embed)
library(ROCR)
library(OutlierDetection)
source("src/TF_BugFix.R")
use_condaenv('r-tensorflow')
tensorflow::use_session_with_seed(100)

train <- read_csv("train.csv")
Y <- data.frame(SWEETORSOUR = as.logical(unlist(train[,1])))
View(Y)
FLAGS <- flags(
  flag_integer("dense_units1", 500),
  flag_numeric("dropout1", 0.5),
  flag_numeric("lambda1", 0.05),
  flag_numeric("bias_lambda1",0.05),
  flag_integer("dense_units2", 400),
  flag_numeric("dropout2", 0.4),
  flag_numeric("lambda2", 0.04),
  flag_numeric("bias_lambda2",0.04),
  flag_integer("dense_units3", 300),
  flag_numeric("dropout3", 0.3),
  flag_numeric("lambda3", 0.03),
  flag_numeric("bias_lambda3",0.03),
  flag_integer("dense_units4", 200),
  flag_numeric("dropout4", 0.2),
  flag_numeric("lambda4", 0.02),
  flag_numeric("bias_lambda4",0.02),
  flag_integer("dense_units5", 100),
  flag_numeric("dropout5", 0.1),
  flag_numeric("lambda5", 0.01),
  flag_numeric("bias_lambda5",0.01),
  flag_integer("batch",50),
  flag_integer("epochs",100),
  flag_string("activation1", 'swish'),
  flag_string("activation2", 'swish'),
  flag_string("activation3", 'swish'),
  flag_string("activation4", 'swish'),
  flag_string("activation5", 'swish'))

callback <- callback_early_stopping(
  monitor = "val_loss",
  patience = 15,
  restore_best_weights = TRUE)


nn1 <- keras_model_sequential() %>%
  layer_dense(units = FLAGS$dense_units1, activation = FLAGS$activation1, bias_regularizer = regularizer_l2(FLAGS$bias_lambda1), kernel_regularizer = regularizer_l2(FLAGS$lambda1), input_shape = ncol(train)-1) %>%
  layer_dropout(rate = FLAGS$dropout1)%>%
  layer_dense(units = FLAGS$dense_units2, activation = FLAGS$activation2, bias_regularizer = regularizer_l2(FLAGS$bias_lambda2), kernel_regularizer = regularizer_l2(FLAGS$lambda2))%>%
  layer_dropout(rate = FLAGS$dropout2)%>%
  layer_dense(units = FLAGS$dense_units3, activation = FLAGS$activation3, bias_regularizer = regularizer_l2(FLAGS$bias_lambda3), kernel_regularizer = regularizer_l2(FLAGS$lambda3))%>%
  layer_dropout(rate = FLAGS$dropout3)%>%
  layer_dense(units = FLAGS$dense_units4, activation = FLAGS$activation4, bias_regularizer = regularizer_l2(FLAGS$bias_lambda4), kernel_regularizer = regularizer_l2(FLAGS$lambda4))%>%
  layer_dropout(rate = FLAGS$dropout4)%>%
  layer_dense(units = FLAGS$dense_units5, activation = FLAGS$activation5, bias_regularizer = regularizer_l2(FLAGS$bias_lambda5), kernel_regularizer = regularizer_l2(FLAGS$lambda5))%>%
  layer_dropout(rate = FLAGS$dropout5)%>%
  layer_batch_normalization()%>%
  layer_dense(units = 1, activation = 'sigmoid') 


nn1 %>% compile(
  loss = 'binary_crossentropy',
  optimizer = 'adam', 
  metrics = c('accuracy')
)

history2 <- nn1 %>% fit(
  as.matrix(train[,-1]),
  as.matrix(Y),
  batch_size = FLAGS$batch, 
  epochs = FLAGS$epochs,
  callbacks = callback,
  validation_split = 0.1,
  verbose = 2
)


