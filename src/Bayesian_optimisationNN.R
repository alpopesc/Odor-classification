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
library(PerformanceAnalytics)
library(tensorflow)


use_condaenv('r-tensorflow')
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
#data$SWEETORSOUR <- as.numeric(data$SWEETORSOUR)
data$VALENCE.PLEASANTNESS <- NULL
data <- na.omit(data)
i_zeros <- remove_zeros(data)
I_zeros <- names(data) %in% i_zeros
data <- data[!I_zeros]
i_const <- remove_constants(data)
I_const <- names(data) %in% i_const
data <- data[!I_const]

#Removing correlated predictors and scaling
data_cor <- cor(as.matrix(data[,-2])) 
hc <- findCorrelation(data_cor, cutoff=0.99) +1
data <- data[,-c(sort(hc))]
data <- cbind(data[,2], scale(data[,-2]))

#Removing outliers
#outliers <- depthout(data)
#data <- data[-outliers[["Location of Outlier"]],] 


cvFolds <- function(Y_, V){
  # Create CV folds (stratify by outcome)   
  Y0 <- split(sample(which(Y_==0)), rep(1:V, length=length(which(Y_==0))))
  Y1 <- split(sample(which(Y_==1)), rep(1:V, length=length(which(Y_==1))))
  folds <- vector("list", length=V)
  for (v in seq(V)) {folds[[v]] <- c(Y0[[v]], Y1[[v]])}     
  return(folds)
}


CV_AUC <- function(units_1, units_2, units_3,dropout_1, dropout_2,lso1,lso2,lso3, epos, i_data = data, V_fold = 5){
  folds <- cvFolds(i_data[,1], V_fold)
  Predictions <- matrix()
  Predictions <- NULL
  Labels <- matrix()
  Labels <- NULL
  for(i in 1:V_fold){
    idc <- sample(folds[[i]], min(mapply(length, folds)))
    train_set <- i_data[-idc,]
    val_set <- i_data[idc,]

    
    nn1 <- keras_model_sequential() %>%
      layer_dense(units = units_1, activation = 'swish', bias_regularizer = regularizer_l2(lso1), kernel_regularizer = regularizer_l2(lso1), input_shape = ncol(train_set)-1) %>%
      layer_dropout(rate = dropout_1)%>%
      layer_dense(units = units_2, activation = 'swish', bias_regularizer = regularizer_l2(lso2), kernel_regularizer = regularizer_l2(lso2))%>%
      layer_dropout(rate = dropout_2)%>%
      layer_dense(units = units_3, activation = 'swish', bias_regularizer = regularizer_l2(lso3), kernel_regularizer = regularizer_l2(lso3))%>%
      layer_batch_normalization()%>%
      layer_dense(units = 1, activation = 'sigmoid') 
    

    
    nn1 %>% compile(
      loss = 'binary_crossentropy',
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


#Examples of previously used model after it was decided to reduces flexibilty of NN
cv1 <- CV_AUC(30, 15, 10, 0.4, 0.4, 0.4, 0.4, 0.4, 56, data, 10) 
cv2 <- CV_AUC(36, 23, 39, 0.393, 0.494, 0.3624, 0.2483, 0, 56, data, 10)



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
                                            init_points = 5, init_grid_dt = search_grid, 
                                            n_iter = 100, acq = "ucb")


