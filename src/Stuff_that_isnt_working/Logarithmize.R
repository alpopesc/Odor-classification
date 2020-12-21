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

use_condaenv('r-tensorflow')
tensorflow::tf$random$set_seed(55)
set.seed(55)



get_skewed_names <- function(d){
  ret <- c()
  ret <- NULL
  for (i in colnames(d[,c(-1,-2)])) {
    if (abs(skewness(d[[i]], type = 2)) > 0.7) {
      ret <- append(ret, i)
    }
  }
  ret
}

return_logd_data <- function(d, skw_names){
  for(i in skw_names){
    if (0 %in% d[[i]]) {
      for (j in 1:nrow(d)) {
        if (d[j,i] > 0) {
          d[j,i] <- log(d[j,i])
        }
      }
    } else {
      d[[i]] <- log(d[[i]])
    }
  }
  return(d)
}