library(readr)
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

