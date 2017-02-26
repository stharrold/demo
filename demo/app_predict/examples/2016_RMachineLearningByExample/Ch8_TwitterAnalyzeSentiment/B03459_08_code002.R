############################################################################
# Chapter      :   8 
# Description  :   Sentiment Classification of tweets using SVM
############################################################################

library(e1071) 
library(caret) 
library(kernlab) 
library(ROCR) 
library(RTextTools)
source("performance_plot_utils.R")

# load labeled dataset
labeledDSFilePath = "labeled_tweets.csv"
labeledDataset = read.csv(labeledDSFilePath, header = FALSE)

# transform polarity labels
labeledDataset$V1 = sapply(labeledDataset$V1, 
		function(x) 
			if(x==4) 
				x <- "positive" 
			else if(x==0) 
				x<-"negative" 
			else x<- "none")

#select required columns only
requiredColumns <- c("V1","V6")

# extract only positive/negative labeled tweets 
tweets<-as.matrix(labeledDataset[labeledDataset$V1 
			%in% c("positive","negative")
			,requiredColumns])

indexes <- createDataPartition(tweets[,1], p=0.7, list = FALSE)

train.data <- tweets[indexes,]
test.data <- tweets[-indexes,]


train.dtMatrix <- create_matrix(train.data[,2], 
                        language="english" , 
                        removeStopwords=TRUE, 
                        removeNumbers=TRUE,
                        stemWords=TRUE,
                        weighting = tm::weightTfIdf)
						

# library issue with weight assignment
# Quick hack => Steps to correct:
# 1. trace("create_matrix",edit=T)
# Line 42: Acronym => acronym

# Alternate method if you want to check: use tm - DocumentTermMatrix
# link: http://stackoverflow.com/questions/16630627/recreate-same-document-term-matrix-with-new-data
# important parameter is dictionary so tf-idf matrix for test data has
# same dimensions as training data
						
test.dtMatrix <- create_matrix(test.data[,2], 
                               language="english" , 
                               removeStopwords=TRUE, 
                               removeNumbers=TRUE,
                               stemWords=TRUE,
                               weighting = tm::weightTfIdf,
                               originalMatrix=train.dtMatrix)

test.data.size <- nrow(test.data)


svm.model <- svm(train.dtMatrix, as.factor(train.data[,1]))

## view inital model details
summary(svm.model)

## predict and evaluate results
svm.predictions <- predict(svm.model, test.dtMatrix)

true.labels <- as.factor(test.data[,1])

confusionMatrix(data=svm.predictions, reference=true.labels, positive="positive")




## hyperparameter optimizations

# run grid search
cost.weights <- c(0.1, 10, 100)
gamma.weights <- c(0.01, 0.25, 0.5, 1)
tuning.results <- tune(svm, train.dtMatrix, as.factor(train.data[,1]), kernel="radial", 
                       ranges=list(cost=cost.weights, gamma=gamma.weights))

# view optimization results
print(tuning.results)

# plot results
plot(tuning.results, cex.main=0.6, cex.lab=0.8,xaxs="i", yaxs="i")

# get best model and evaluate predictions
svm.model.best = tuning.results$best.model
svm.predictions.best <- predict(svm.model.best, test.dtMatrix)
confusionMatrix(data=svm.predictions.best, reference=true.labels, positive="positive")


# plot best model evaluation metric curves
svm.predictions.best <- predict(svm.model.best
							, test.dtMatrix, decision.values = T)

svm.prediction.values <- attributes(svm.predictions.best)$decision.values

predictions <- prediction(svm.prediction.values, true.labels)

par(mfrow=c(1,2))
plot.roc.curve(predictions, title.text="SVM ROC Curve")
plot.pr.curve(predictions, title.text="SVM Precision/Recall Curve")
