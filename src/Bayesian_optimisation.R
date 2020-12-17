library(keras)
library(readr)
library(purrr)
library(dplyr)
library(reticulate)
library(embed)
library(ROCR)
library(OutlierDetection)
library(caret)
library(rBayesianOptimization)
library(cvAUC)

use_condaenv('r-tensorflow')
tensorflow::use_session_with_seed(100)

import_data <- data

cvFolds <- function(Y_, V){
  # Create CV folds (stratify by outcome)   
  Y0 <- split(sample(which(Y_==0)), rep(1:V, length=length(which(Y_==0))))
  Y1 <- split(sample(which(Y_==1)), rep(1:V, length=length(which(Y_==1))))
  folds <- vector("list", length=V)
  for (v in seq(V)) {folds[[v]] <- c(Y0[[v]], Y1[[v]])}     
  return(folds)
}

CV_AUC <- function(units_1, units_2, units_3,dropout_1, dropout_2,lso1,lso2,lso3, epos, data = import_data, V_fold = 5){
  folds <- cvFolds(data, V_fold)
  Predictions <- matrix()
  Predictions <- NULL
  Labels <- matrix()
  Labels <- NULL
  for(i in 1:V_fold){
    idc <- sample(x[[i]], min(mapply(length, x)))
    train_set <- data[-idc,]
    val_set <- data[idc,]
    
    nn1 <- keras_model_sequential() %>%
      layer_dense(units = units_1, activation = 'swish', bias_regularizer = regularizer_l2(lso1), kernel_regularizer = regularizer_l2(lso1), input_shape = ncol(train_set)-1) %>%
      layer_dropout(rate = dropout_1)%>%
      layer_dense(units = units_2, activation = 'swish', bias_regularizer = regularizer_l2(lso2), kernel_regularizer = regularizer_l2(lso2))%>%
      layer_dropout(rate = dropout_2)%>%
      layer_dense(units = units_3, activation = 'swish', bias_regularizer = regularizer_l2(lso3), kernel_regularizer = regularizer_l2(lso3))%>%
      layer_batch_normalization()%>%
      layer_dense(units = 1, activation = 'sigmoid') 
    
    
    nn1 %>% compile(
      loss = loss,
      optimizer = 'adam', # stochastic gradient descent
      metrics = c('accuracy')
    )
    
    history2 <- nn1 %>% fit(
      as.matrix(train_set[,-1]),
      as.logical(train_set[,1]),
      batch_size = nrow(train_set), 
      epochs = epos,
      #callbacks = callback,
      validation_split = 0,
      verbose = 0,
      class_weight = list("0"=0.82,"1"=1.283)
    )
    
    Predictions <- cbind(Predictions, predict(nn1, as.matrix(val_set[,-1])))
    Labels <- cbind(Labels, as.matrix(val_set[,1]))
  }
  
  return(list(Score = cvAUC(Predictions,Labels)[["cvAUC"]], Pred = 0))
}

cv <- CV_AUC(30, 30, 30, 0.4, 0.4, 0.5, 0.5, 0.5, 150, data, 5)

#Boundries
search_bound <- list(units_1 = c(5,50),
                     units_2 = c(5,50),
                     units_3 = c(5,50),
                     dropout_1 = c(0, 0.7),
                     dropout_2 = c(0, 0.7),
                     lso1 = c(0,0.5),
                     lso2 = c(0,0.5),
                     lso3 = c(0,0.5),
                     epos = c(50,300))

#Initial values
search_grid <- data.frame(units_1 = sample(5:50,10),
                          units_2 = sample(5:50,10),
                          units_3 = sample(5:50,10),
                          dropout_1 = runif(10, 0, 0.7),
                          dropout_2 = runif(10, 0, 0.7),
                          lso1 = runif(10,0,0.5),
                          lso2 = runif(10,0,0.5),
                          lso3 = runif(10,0,0.5),
                          epos = sample(50:300,10))
                          

all_about_the_bayes <- BayesianOptimization(FUN = CV_AUC, bounds = search_bound, 
                                            init_points = 10, init_grid_dt = search_grid, 
                                            n_iter = 10, acq = "ucb")

