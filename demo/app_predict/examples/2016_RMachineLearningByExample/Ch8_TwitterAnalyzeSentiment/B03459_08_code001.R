############################################################################
# Chapter      :   8 
# Description  :   Polarity analysis of Twitter data
############################################################################

library(twitteR)
library(stringr)
library(tm)
library(ggplot2)
consumerSecret = ""
consumerKey = ""

setup_twitter_oauth(consumer_key = consumerKey,consumer_secret = consumerSecret)

# list of positive/negative words from opinion lexicon
pos.words = scan(file= 'positive-words.txt', what='character', comment.char=';')
neg.words = scan(file= 'negative-words.txt', what='character', comment.char=';')


#extract search tweets
extractTweets <- function(searchTerm,tweetCount){
  # search term tweets
  tweets = searchTwitter(searchTerm,n=tweetCount)
  tweets.df = twListToDF(tweets)
  tweets.df$text <- sapply(tweets.df$text,function(x) iconv(x,to='UTF-8'))
  
  return(tweets.df)
}

#extract timeline tweets
extractTimelineTweets <- function(username,tweetCount){
  # timeline tweets
  twitterUser <- getUser(username)
  tweets = userTimeline(twitterUser,n=tweetCount)
  tweets.df = twListToDF(tweets)
  tweets.df$text <- sapply(tweets.df$text,function(x) iconv(x,to='UTF-8'))
  
  return(tweets.df)
}


# clean and transform tweets
transformTweets <- function(tweetDF){
  tweetCorpus <- Corpus(VectorSource(tweetDF$text))
  tweetCorpus <- tm_map(tweetCorpus, tolower)
  tweetCorpus <- tm_map(tweetCorpus, removePunctuation)
  tweetCorpus <- tm_map(tweetCorpus, removeNumbers)
  
  # remove URLs
  removeURL <- function(x) gsub("http://[[:alnum:]]*", "", x)
  tweetCorpus <- tm_map(tweetCorpus, removeURL) 
  
  # remove stop words
  twtrStopWords <- c(stopwords("english"),'rt','http','https')
  tweetCorpus <- tm_map(tweetCorpus, removeWords, twtrStopWords)
  
  tweetCorpus <- tm_map(tweetCorpus, PlainTextDocument)
  
  #convert back to dataframe
  tweetDataframe<-data.frame(text=unlist(sapply(tweetCorpus, `[`, "content")), 
                             stringsAsFactors=F)
  
  #split each doc into words
  splitText <- function(x) {
    word.list = str_split(x, '\\s+')
    words = unlist(word.list)
  }
  
  tweetDataframe$wordList = sapply(tweetDataframe$text,function(text) splitText(text))
  
  return (tweetDataframe)
}


#score the words
scoreTweet <- function(wordList) {
    # compare our words to the dictionaries of positive & negative terms
    pos.matches = match(wordList, pos.words)
    neg.matches = match(wordList, neg.words)
    
    # match() returns the position of the matched term or NA
    # we just want a TRUE/FALSE:
    pos.matches = !is.na(pos.matches)
    neg.matches = !is.na(neg.matches)
    
    # and conveniently enough, TRUE/FALSE will be treated as 1/0 by sum():
    score = sum(pos.matches) - sum(neg.matches)
	return(score)
  }

  
analyzeUserSentiments <- function(search,tweetCount){ 
  
  #extract tweets
  tweetsDF <- extractTimelineTweets(search,tweetCount) #extractTweets(search,tweetCount)
  
  # transformations
  transformedTweetsDF <- transformTweets(tweetsDF)
  
  #score the words  
  transformedTweetsDF$sentiScore = sapply(transformedTweetsDF$wordList,function(wordList) scoreTweet(wordList))
  transformedTweetsDF$search <- search
  
  return(transformedTweetsDF) 
}

analyzeTrendSentiments <- function(search,tweetCount){ 
  
  #extract tweets
  tweetsDF <- extractTweets(search,tweetCount)
  
  # transformations
  transformedTweetsDF <- transformTweets(tweetsDF)
  
  #score the words  
  transformedTweetsDF$sentiScore = sapply(transformedTweetsDF$wordList,function(wordList) scoreTweet(wordList))
  transformedTweetsDF$search <- search
  
  return(transformedTweetsDF) 
}

makeInIndiaSentiments <- analyzeTrendSentiments("makeinindia",1500)
qplot(makeInIndiaSentiments$sentiScore) + theme_grey(base_size = 18) 


user1Timeline <- analyzeUserSentiments("Number10gov",1500)
user2Timeline <- analyzeUserSentiments("POTUS",1500)
user3Timeline <- analyzeUserSentiments("PMOIndia",1500)
all.scores <- rbind(user1Timeline,user2Timeline,user3Timeline)
ggplot(data=all.scores)+geom_histogram(mapping=aes(x=sentiScore,fill=search),binwidth = 1)+ facet_grid(search~.)+scale_fill_grey()