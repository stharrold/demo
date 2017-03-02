############################################################################
# Chapter      :   8 
# Description  :   Sentiment Classification of tweets using Boosting
############################################################################

library(caret) # model training\optimizations
library(RTextTools)
source("performance_plot_utils.R") # plot model metrics

# load labeled dataset
labeledDSFilePath = "labeled_tweets.csv"
labeledDataset = read.csv(labeledDSFilePath, header = FALSE)

# transform polarity labels
labeledDataset$V1 = sapply(labeledDataset$V1, function(x) if(x==4) x <- "positive" else if(x==0) x<-"negative" else x<- "none")

# extract only positive/negative labeled tweets with required columns (polarity and tweet text)
requiredColumns <- c("V1","V6")
tweets<-as.matrix(labeledDataset[labeledDataset$V1 %in% c("positive","negative"),requiredColumns])

indexes <- createDataPartition(tweets[,1], p=0.7, list = FALSE)

train.data <- tweets[indexes,]
test.data <- tweets[-indexes,]


train.dtMatrix <- create_matrix(train.data[,2], 
                        language="english" , 
                        removeStopwords=TRUE, 
                        removeNumbers=TRUE,
                        stemWords=TRUE,
                        weighting = tm::weightTfIdf)
						

						
test.dtMatrix <- create_matrix(test.data[,2], 
                               language="english" , 
                               removeStopwords=TRUE, 
                               removeNumbers=TRUE,
                               stemWords=TRUE,
                               weighting = tm::weightTfIdf,
                               originalMatrix=train.dtMatrix)

test.data.size <- nrow(test.data)


train.container <- create_container(train.dtMatrix, 
                              as.factor(train.data[,1]), 
                              trainSize=1:nrow(train.data), 
                              virgin=FALSE)
							  
test.dtMatrix <- create_matrix(test.data[,2], 
                               language="english" , 
                               removeStopwords=TRUE, 
                               removeNumbers=TRUE,
                               stemWords=TRUE,
                               weighting = tm::weightTfIdf,
                               originalMatrix=train.dtMatrix)

test.data.size <- nrow(test.data)
test.container <- create_container(test.dtMatrix, 
                                        labels=rep(0,test.data.size), 
                                        testSize=1:test.data.size, 
                                        virgin=FALSE)

							  
							  
boosting.model <- train_model(train.container, "BOOSTING"
						, maxitboost=500)
boosting.classify <- classify_model(test.container, boosting.model)

predicted.labels <- boosting.classify[,1]
true.labels <- as.factor(test.data[,1])

confusionMatrix(data = predicted.labels, 
                reference = true.labels, 
                positive = "positive")
				
	
# plot model evaluation metric curves
predictions <- prediction(boosting.classify[,"LOGITBOOST_PROB"]
					, true.labels
					,label.ordering =c("negative","positive"))
					
par(mfrow=c(1,2))
plot.roc.curve(predictions, title.text="Boosting ROC Curve")
plot.pr.curve(predictions, title.text="Boosting Precision/Recall Curve")


# Cross validation
N=10
set.seed(42)
cross_validate(train.container,N,"BOOSTING"
		, maxitboost=500)
