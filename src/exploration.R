library(readr)
<<<<<<< HEAD
library(ggplot2)
library(purrr)
library(pheatmap)
library(dplyr)
library(Rtsne)
=======
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

d <- remove_zeros(data_t)
d$Intensity <- as.numeric(d$Intensity == "high")
d$SWEETORSOUR <- as.numeric(d$SWEETORSOUR)
View(d)

#It would be good to code a function that removes outliers from this high_dimensional data

data_types(d)
#With the previous command we can see that except for the columns at the beginning there are only numerical columns(predcitors)
#Now we have to see if have to include the intesity in our model.

int_sour <- d[,c(1,3)]
int_sour$Intensity <- as.logical(int_sour$Intensity == 1)
View(int_sour)
tf <- table(int_sour)
tf
tf[,1] <- tf[,1]/sum(tf[,1])
tf[,2] <- tf[,2]/sum(tf[,2])
tf
barplot(t(tf))
barplot(tf)

#we see the data is slightly imbalanced 

library(pheatmap)
predictor_matrix <- remove_constants(d[,-3])
View(predictor_matrix)
correlation_matrix <- as.matrix((cor(predictor_matrix, use = "complete.obs")))
anyNA(correlation_matrix)
View(correlation_matrix)
#pheatmap(correlation_matrix, legend = T)



#Creating PCA
#Note that it is not necessary to normalize the data we will suppose that the measurement of each sample ist the same

predictor_matrix_sc <- scale(predictor_matrix)
pca <- prcomp(predictor_matrix_sc, scale = T, center = T)
pca.data <- data.frame(X=pca$x[,1], Y=pca$x[,2])
plot(pca.data)



## make a scree plot
pca.var <- pca$sdev^2
pca.var.per <- round(pca.var/sum(pca.var)*100, 1)
barplot(pca.var.per[1:50], main="Scree Plot", xlab="Principal Component", ylab="Variation")

library(ggplot2)
cl <- kmeans(data.frame(pca$x), centers=30, nstart=500)
p <-ggplot(pca.data, aes(x = X, y = Y, col = cl$cluster) ) +
  geom_point() +
  xlab(paste("PC1 - ", pca.var.per[1], "%", sep="")) +
  ylab(paste("PC2 - ", pca.var.per[2], "%", sep="")) 
p + scale_color_gradient(low="green", high="red")



library(dplyr)
o <- order(d$SWEETORSOUR)
pca_col <- c(rep(2,nrow(d)), rep(3,nrow(d)))
plot(pca.data[o,], col= pca_col[d$SWEETORSOUR[o] == 1])
View(d)





library(Rtsne)
tsne <- Rtsne(predictor_matrix,check_duplicates = F)
plot(tsne$Y[o,], col = pca_col[d$SWEETORSOUR[o] == 1])




normality_pval <- function(de){
    shapiro.test(de)$p.value
}

normality_score <- function(data){
  c <- 0
  for(i in colnames(data)){
    p <- normality_pval(unlist(data[i]))
    if(p > 0.05){
      c <- c + 1
    }
  }
  c/ncol(data)
}

#We can see that most of the data is not really normally distributed so we should not consider LDA QDA.

some_matrix <- predictor_matrix
data_types(some_matrix)
normality_score(some_matrix)
>>>>>>> 85048560c8bf75f6fb4de74dd0dda1ad30099e46

training_data <- read_csv("data/training_data.csv")
View(training_data)
data_t <- training_data
str(data_t)
ncol(data_t)
typeof(data_t)

#Function to remove Zero columns
remove_zeros <- function(data){
  vec <- c()
  for(i in colnames(data)){
    if(all(data[i] == 0)){
      data[i] <- NULL
    }
  }
  data
}


#Checking all the different datatypes
data_types <- function(frame) {
  res <- lapply(frame, class)
  res_frame <- data.frame(unlist(res))
  barplot(table(res_frame), main="Data Types", col="steelblue", ylab="Number of Features")
  c(sum(res_frame == "numeric"), sum(res_frame != "numeric"))
}


#Function to remove constant columns
remove_constants <-function(data){
  for(i in colnames(data)){
    if(sd(unlist(data[i])) == 0){
      data[i] <- NULL
    }
  }
  data
}

d <- remove_zeros(data_t)
d$Intensity <- as.numeric(d$Intensity == "high")
d$SWEETORSOUR <- as.numeric(d$SWEETORSOUR)
View(d)


data_types(d)
#With the previous command we can see that except for the columns at the beginning there are only numerical columns(predcitors)

#Check for inbalances in data
SorS <- as.logical(d$SWEETORSOUR)
tf <- table(SorS)
barplot(tf)


# Correlation heatmap 
predictor_matrix <- remove_constants(d[,-3])
View(predictor_matrix)
correlation_matrix <- as.matrix((cor(predictor_matrix, use = "complete.obs")))
anyNA(correlation_matrix)
View(correlation_matrix)
#pheatmap(correlation_matrix, legend = T) ### Takes a long of time therefore it is commented.



#Creating PCA
predictor_matrix_sc <- scale(predictor_matrix)
pca <- prcomp(predictor_matrix_sc, scale = T, center = T)
pca.data <- data.frame(X=pca$x[,1], Y=pca$x[,2])
plot(pca.data)



# Scree plot for PCA
pca.var <- pca$sdev^2
pca.var.per <- round(pca.var/sum(pca.var)*100, 1)
barplot(pca.var.per[1:50], main="Scree Plot", xlab="Principal Component", ylab="Variation")


#PCA with K-means
cl <- kmeans(data.frame(pca$x), centers=30, nstart=500)
p <-ggplot(pca.data, aes(x = X, y = Y, col = cl$cluster) ) +
  geom_point() +
  xlab(paste("PC1 - ", pca.var.per[1], "%", sep="")) +
  ylab(paste("PC2 - ", pca.var.per[2], "%", sep="")) 
p + scale_color_gradient(low="green", high="red")



#PCA with color according to SWEETORSOUR
o <- order(d$SWEETORSOUR)
pca_col <- c(rep(2,nrow(d)), rep(3,nrow(d)))
plot(pca.data[o,], col= pca_col[d$SWEETORSOUR[o] == 1])
View(d)


#t-SNE
tsne <- Rtsne(predictor_matrix,check_duplicates = F)
plot(tsne$Y[o,], col = pca_col[d$SWEETORSOUR[o] == 1])


#Test to see how normal the coulumns in our data is
normality_pval <- function(de){
    shapiro.test(de)$p.value
}

normality_score <- function(data){
  c <- 0
  for(i in colnames(data)){
    p <- normality_pval(unlist(data[i]))
    if(p > 0.05){
      c <- c + 1
    }
  }
  c/ncol(data)
}

some_matrix <- predictor_matrix
data_types(some_matrix)
normality_score(some_matrix)

# In  nutshell wee see that we have highly dimensional data that does not form clear clusters  correspnding
# to SWEETSOUR. Additionaly a lot of the predictors seem to correlate with each other and the data 
# seems to be skewed and inbalanced generally. 


