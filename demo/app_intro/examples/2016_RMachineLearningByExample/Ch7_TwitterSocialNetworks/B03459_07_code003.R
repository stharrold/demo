############################################################################
# Chapter      :   7 
# Description  :   Topic Modeling using LDA
# Note         :   Your results may be different than the ones discussed 
#                  in the chapter due to dyanmic nature of Twitter
############################################################################
library(twitteR)
library(tm)
library(stringr)
library(topicmodels)
library(ggplot2)

consumerSecret = "YOUR CONSUMER SECRET"
consumerKey = "YOUR CONSUMER KEY"

setup_twitter_oauth(consumer_key = consumerKey,consumer_secret = consumerSecret)
atISS <- getUser("ISS_Research")

# extract iss_research tweets
tweets <- userTimeline(atISS, n = 1000)
tweets.df=twListToDF(tweets)
tweets.df$text <- sapply(tweets.df$text,function(x) iconv(x,to='UTF-8'))


#transformations
twtrCorpus <- Corpus(VectorSource(tweets.df$text))
twtrCorpus <- tm_map(twtrCorpus, tolower)
twtrCorpus <- tm_map(twtrCorpus, removePunctuation)
twtrCorpus <- tm_map(twtrCorpus, removeNumbers)
myStopwords <- c(stopwords("english"), "available", "via","amp","space","outerspace","spacestation","issresearch","nasa","science")
twtrCorpus <- tm_map(twtrCorpus, removeWords, myStopwords)
twtrCorpus <- tm_map(twtrCorpus, PlainTextDocument)


#DTM
twtrDTM <- DocumentTermMatrix(twtrCorpus, control = list(minWordLength = 1))


#topic modeling

# find 8 topics
ldaTopics <- LDA(twtrDTM, k = 8) 

#first 6 terms of every topic
ldaTerms <- terms(ldaTopics, 6) 

# concatenate terms
(ldaTerms <- apply(ldaTerms, MARGIN = 2, paste, collapse = ", "))

# first topic identified for every tweet
firstTopic <- topics(ldaTopics, 1)

topics <- data.frame(date=as.Date(tweets.df$created), firstTopic)
qplot(date, ..count.., data=topics, geom="density",fill=ldaTerms[firstTopic], position="stack")+scale_fill_grey()

