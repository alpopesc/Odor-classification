library(readr)
training_data <- read_csv("data/training_data.csv")
View(training_data)
data_t <- training_data
str(data_t)
ncol(data_t)
typeof(data_t)

remove_zeros <- function(data){
  vec <- c()
  for(i in colnames(data)){
    if(all(data[i] == 0)){
      data[i] <- NULL
    }
  }
  data
}


d <- remove_zeros(data_t)
ncol(d)

#checking all the different datatypes

data_types <- function(frame) {
  res <- lapply(frame, class)
  res_frame <- data.frame(unlist(res))
  barplot(table(res_frame), main="Data Types", col="steelblue", ylab="Number of Features")
  c(sum(res_frame == "numeric"), sum(res_frame != "numeric"))
}

remove_NA <- function(data){
  for(i in 1:(ncol(data)-1)){
    if((is.na(data[,i]))){
      data <- data[,-i]
    }
    if((is.na(data[i,]))){
      data <- data[-i,]
    }
  }
  data
}


data_types(d)
d <- na.omit(d)

library(pheatmap)
predictor_matrix <- d[,4:ncol(d)]
correlation_matrix <- as.data.frame((cor(predictor_matrix, use = "complete.obs")))
correlation_matrix <- remove_NA(correlation_matrix)
anyNA(correlation_matrix)
View(correlation_matrix)
heatmap(correlation_matrix)




