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

data_types(d)
#With the previous command we can see that except for the columns at the beginning there are only numerical columns(predcitors)
#Now we have to see if have to include the intesity in our model.

int_sour <- d[,c(1,3)]
tf <- table(int_sour)
tf[,1] <- tf[,1]/sum(tf[,1])
tf[,2] <- tf[,2]/sum(tf[,2])
barplot(t(tf))
barplot(tf)

 

library(pheatmap)
d <- remove_zeros(data_t)
predictor_matrix <- remove_constants(d[,4:ncol(d)])
View(predictor_matrix)
sd(unlist(predictor_matrix[,352]))
correlation_matrix <- as.matrix((cor(predictor_matrix, use = "complete.obs")))
anyNA(correlation_matrix)
View(correlation_matrix)
heatmap(correlation_matrix)



#Creating PCA
#Note that it is not necessary to normalize the data we will suppose that the measurement of each sample ist the same

predictor_matrix_sc <- scale(predictor_matrix)
pca <- prcomp(predictor_matrix_sc, scale = F, center = T)
pca.data <- data.frame(X=pca$x[,1], Y=pca$x[,2])
plot(pca.data)



## make a scree plot
pca.var <- pca$sdev^2
pca.var.per <- round(pca.var/sum(pca.var)*100, 1)
barplot(pca.var.per[1:50], main="Scree Plot", xlab="Principal Component", ylab="Variation")

library(ggplot2)
cl <- kmeans(data.frame(pca$x), centers=3, nstart=50)
p <-ggplot(pca.data, aes(x = X, y = Y, col = cl$cluster) ) +
  geom_point() +
  xlab(paste("PC1 - ", pca.var.per[1], "%", sep="")) +
  ylab(paste("PC2 - ", pca.var.per[2], "%", sep="")) 
p + scale_color_gradient(low="blue", high="yellow")










