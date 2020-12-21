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
library(bestNormalize)
library(kernlab)
library(tidymodels)
library(themis)

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
hc <- findCorrelation(data_cor, cutoff=0.96) + 1
data <- data[,-c(sort(hc))]
data <- cbind(data[,2], scale(data[,-2]))

#Transforming SWEETORSOUR column to categorical
data$SWEETORSOUR <- as.factor(as.logical(data$SWEETORSOUR))



View(data)



folds <- vfold_cv(data, v = 10)
View(data)
data_pre_proc <-
  recipe(SWEETORSOUR ~ ., data = data)

svm_mod <-
  svm_rbf(mode = "classification", cost = tune(), rbf_sigma = tune()) %>%
  set_engine("kernlab")

svm_wflow <-
  workflow() %>%
  add_model(svm_mod) %>%
  add_recipe(data_pre_proc)

svm_set <- parameters(svm_wflow)
svm_set

svm_set <- 
  svm_set %>% 
  update(num_comp = num_comp(c(0L, 20L)))

set.seed(55)
search_res <-
  svm_wflow %>% 
  tune_bayes(
    resamples = folds,
    # To use non-default parameter ranges
    param_info = svm_set,
    # Generate five at semi-random to start
    initial = 5,
    iter = 200,
    # How to measure performance
    metrics = metric_set(roc_auc),
    control = control_bayes(no_improve = 40, verbose = TRUE)
  )

