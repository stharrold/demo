############################################################################
# Chapter      :   7 
# Description  :   Mine Tweets to:
#                   a) Find most frequently used words
#                   b) Find associations
#                   c) Cluster terms
# Viz          :   Bar graphs, word clouds, dendograms
# Note         :   Your results may be different than the ones discussed 
#                  in the chapter due to dyanmic nature of Twitter
############################################################################


library(twitteR)
library(ggplot2)
library(stringr)
library(tm)
library(wordcloud)



consumerSecret = "YOUR CONSUMER SECRET"
consumerKey = "YOUR CONSUMER KEY"

setup_twitter_oauth(consumer_key = consumerKey,consumer_secret = consumerSecret)

# trending tweets
trendingTweets = searchTwitter("#ResolutionsFor2016",n=1000)
trendingTweets.df = twListToDF(trendingTweets)
trendingTweets.df$text <- sapply(trendingTweets.df$text,function(x) iconv(x,to='UTF-8'))

# encode tweet source as iPhone, iPad, Android or Web
enodeSource <- function(x) {
  if(x=="<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>"){
    gsub("<a href=\"http://twitter.com/download/iphone\" rel=\"nofollow\">Twitter for iPhone</a>", "iPhone", x,fixed=TRUE)
  }else if(x=="<a href=\"http://twitter.com/#!/download/ipad\" rel=\"nofollow\">Twitter for iPad</a>"){
    gsub("<a href=\"http://twitter.com/#!/download/ipad\" rel=\"nofollow\">Twitter for iPad</a>","iPad",x,fixed=TRUE)
  }else if(x=="<a href=\"http://twitter.com/download/android\" rel=\"nofollow\">Twitter for Android</a>"){
    gsub("<a href=\"http://twitter.com/download/android\" rel=\"nofollow\">Twitter for Android</a>","Android",x,fixed=TRUE)
  } else if(x=="<a href=\"http://twitter.com\" rel=\"nofollow\">Twitter Web Client</a>"){
    gsub("<a href=\"http://twitter.com\" rel=\"nofollow\">Twitter Web Client</a>","Web",x,fixed=TRUE)
  } else if(x=="<a href=\"http://www.twitter.com\" rel=\"nofollow\">Twitter for Windows Phone</a>"){
    gsub("<a href=\"http://www.twitter.com\" rel=\"nofollow\">Twitter for Windows Phone</a>","Windows Phone",x,fixed=TRUE)
  }else {
    x
  }
}

trendingTweets.df$tweetSource = sapply(trendingTweets.df$statusSource,function(sourceSystem) enodeSource(sourceSystem))

# transformations
tweetCorpus <- Corpus(VectorSource(trendingTweets.df$text))
tweetCorpus <- tm_map(tweetCorpus, tolower)
tweetCorpus <- tm_map(tweetCorpus, removePunctuation)
tweetCorpus <- tm_map(tweetCorpus, removeNumbers)

# remove URLs
removeURL <- function(x) gsub("http[[:alnum:]]*", "", x)
tweetCorpus <- tm_map(tweetCorpus, removeURL) 

# remove stop words
twtrStopWords <- c(stopwords("english"),'resolution','resolutions','resolutionsfor','resolutionsfor2016','2016','new','year','years','newyearresolution')
tweetCorpus <- tm_map(tweetCorpus, removeWords, twtrStopWords)

tweetCorpus <- tm_map(tweetCorpus, PlainTextDocument)

# Term Document Matrix
twtrTermDocMatrix <- TermDocumentMatrix(tweetCorpus, control = list(minWordLength = 1))

# Terms occuring in more than 30 times
which(apply(twtrTermDocMatrix,1,sum)>=30)


# Frequency-Association
(frequentTerms<-findFreqTerms(twtrTermDocMatrix,lowfreq = 10))

term.freq <- rowSums(as.matrix(twtrTermDocMatrix))
# picking only a subset
subsetterm.freq <- subset(term.freq, term.freq >= 10)

# create data frames
frequentTermsSubsetDF <- data.frame(term = names(subsetterm.freq), freq = subsetterm.freq)
frequentTermsDF <- data.frame(term = names(term.freq), freq = term.freq)

# sort by frequency
frequentTermsSubsetDF <- frequentTermsSubsetDF[with(frequentTermsSubsetDF, order(-frequentTermsSubsetDF$freq)), ]
frequentTermsDF <- frequentTermsDF[with(frequentTermsDF, order(-frequentTermsDF$freq)), ]

# words by frequency
ggplot(frequentTermsSubsetDF, aes(x = reorder(term,freq), y = freq)) + geom_bar(stat = "identity") +xlab("Terms") + ylab("Frequency") + coord_flip()

# wordcloud
wordcloud(words=frequentTermsDF$term, freq=frequentTermsDF$freq,random.order=FALSE)


# top retweet
head(subset(trendingTweets.df$text, grepl("trillionaire",trendingTweets.df$text) ),n=1)


# Associatons
(fitness.associations <- findAssocs(twtrTermDocMatrix,"fitness",0.25))
fitnessTerm.freq <- rowSums(as.matrix(fitness.associations$fitness))
fitnessDF <- data.frame(term=names(fitnessTerm.freq),freq=fitnessTerm.freq)
fitnessDF <- fitnessDF[with(fitnessDF, order(-fitnessDF$freq)), ]
ggplot(fitnessDF,aes(x=reorder(term,freq),y=freq))+geom_bar(stat = "identity") +xlab("Terms") + ylab("Associations") + coord_flip()



# Source by retweet count
trendingTweetsSubset.df <- subset(trendingTweets.df, trendingTweets.df$retweetCount >= 5000 )
ggplot(trendingTweetsSubset.df, aes(x = tweetSource, y = retweetCount/100)) + geom_bar(stat = "identity") +xlab("Source") + ylab("Retweet Count") 



# clustering
twtrTermDocMatrix2 <- removeSparseTerms(twtrTermDocMatrix, sparse = 0.98)

tweet_matrix <- as.matrix(twtrTermDocMatrix2)

# cluster terms
distMatrix <- dist(scale(tweet_matrix))

fit <- hclust(distMatrix,method="single")
plot(fit)
#rect.hclust(fit, k = 6) # cut tree into 6

