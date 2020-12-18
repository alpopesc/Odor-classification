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
library(GA)

use_condaenv('r-tensorflow')
tensorflow::tf$random$set_seed(55)
set.seed(55)


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
data <- cbind(data[,c(1,2)],scale(data[,c(-1,-2)]))
anyNA(data)
data <-data[ , colSums(is.na(data)) == 0]
anyNA(data)


#Removing correlated predictors
data_cor <- cor(as.matrix(data[,-2]))
hc <- findCorrelation(data_cor, cutoff=0.96) +1
data <- data[,-c(sort(hc))]


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

V_fold <- 10

CV_AUC <- function(x){
  gamma_val <- x[2]
  c_val <- x[1]
  folds <- cvFolds(data[,2], V_fold)
  Predictions <- matrix()
  Predictions <- NULL
  Labels <- matrix()
  Labels <- NULL
  for(i in 1:V_fold){
    idc <- sample(folds[[i]], min(mapply(length, folds)))
    train_set <- data[-idc,]
    val_set <- data[idc,]
    
    model <- svm( 
      SWEETORSOUR ~ ., 
      data = train_set, 
      cost = c_val, 
      gamma = gamma_val, 
      type = "nu-classification", 
      kernel = "radial",
      class_weight = list("0"=0.82,"1"=1.283)
    )
    Predictions <- cbind(Predictions, predict(model, as.matrix(val_set[,-2])))
    Labels <- cbind(Labels, as.matrix(val_set[,2]))
  }
  
  return(cvAUC(Predictions,Labels)[["cvAUC"]])
}




para_value_min <- c( c = 1e-4, gamma = 1e-3 )
para_value_max <- c( c = 10, gamma = 2 )

## run genetic algorithm
results <- ga( type = "real-valued", 
               fitness = CV_AUC,
               names = names( para_value_min ), 
               lower = para_value_min, 
               upper = para_value_max,
               popSize = 50, 
               maxiter = 100)







