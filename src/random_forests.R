library(tree)
library(readr)
library(ROCR)
library(randomForest)

#Load data

test_data <- read_csv(file.path("data/test_data.csv"))
training_data <- read_csv(file.path("data/training_data.csv"))
#data$SWEETORSOUR <- as.factor(data$SWEETORSOUR)
data <- na.omit(training_data)

#remove constant columns
remove_constants <-function(data) {
  for(i in colnames(data)) {
    if(sd(unlist(data[[i]])) == 0) {
      data[[i]] <- NULL
    }
  }
  data
}

data$VALENCE.PLEASANTNESS <- NULL
data_intensity <- data$Intensity
data_SWEETORSOUR <- data$SWEETORSOUR
data <- remove_constants(data[,3:ncol(data)])

#To remove correlated predictors

# avec cutoff = 0.90 -> 851 variables
# cutoff = 0.99 -> 1828 variables
data_cor <- cor(as.matrix(data[,2:ncol(data)]))
hc <- findCorrelation(data_cor, cutoff=0.99) 
data <- data[,-c(sort(hc))]
data$SWEETORSOUR <- data_SWEETORSOUR
data$Intensity <- data_intensity

# #tell R which columns contain categorical data
# for (i in colnames(data)) {
#   if (class(data[[i]]) != "numeric") {
#     data[[i]] <- as.factor(data[[i]])
#   }
# }

# Splitting the data into training and validation sets

set.seed(101)
idx.train <- sample(nrow(data), nrow(data)*0.7)
data.train <- data[idx.train,]
data.validation <- data[-idx.train,]

#Let's create random forests

#by default ntree = 500
forests <- randomForest(SWEETORSOUR ~ ., data.train, ntree = 100)
plot(forests)
#it seems that error rate is stabilized after roughly 150 trees
predValid <- predict(forests, data.validation, type = "class")
forests.ROCRpred <- prediction(predValid, data.validation$SWEETORSOUR)
forests.ROCRperf <- performance(forests.ROCRpred, 'tpr', 'fpr')
plot(forests.ROCRperf, lwd = 2, col = "blue")
print(paste('AUC of random forests:', performance(forests.ROCRpred, 'auc')@y.values, sep = ' '))
#0.58

#Look at AUC of training set

predTrain <- predict(forests, data.train, type = "class")
forests.ROCRpred.train <- prediction(as.numeric(predTrain), as.numeric(data.train$SWEETORSOUR))
forests.ROCRperf.train <- performance(forests.ROCRpred.train, 'tpr', 'fpr')
plot(forests.ROCRperf.train, lwd = 2, col = "blue")
print(paste('AUC of random forests:', performance(forests.ROCRpred.train, 'auc')@y.values, sep = ' '))

#Let's fit our model on all the training set to make predictions on test_data

#We only keep the predictors used in training_data
for (i in colnames(test_data)) {
  if (!i %in% colnames(data)) {
    test_data[[i]] <- NULL
  }
}
predTest <- predict(forests, test_data, type = "class")
final.pred <- data.frame(cbind(predTest))
final.pred$Id <- 1:68
names(final.pred)[1] <- "SWEETORSOUR"
final.pred <- final.pred[c("Id", "SWEETORSOUR")]
write.table(final.pred, "first_try_RF.csv", row.names=FALSE, sep=",")

