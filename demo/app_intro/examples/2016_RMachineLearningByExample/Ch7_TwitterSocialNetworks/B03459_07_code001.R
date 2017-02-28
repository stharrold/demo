############################################################################
# Chapter      :   7 
# Description  :   Connect to Twitter app and extract sample tweets
# Note         :   Your results may be different than the ones discussed 
#                  in the chapter due to dyanmic nature of Twitter
############################################################################

library(twitteR)
consumerSecret = "YOUR CONSUMER SECRET"
consumerKey = "YOUR CONSUMER KEY"

setup_twitter_oauth(consumer_key = consumerKey,consumer_secret = consumerSecret)
twitterUser <- getUser("jack")

# extract jack's tweets
tweets <- userTimeline(twitterUser, n = 300)

# get tweet attributes
tweets[[1]]$getClass()

# get retweets count
tweets[[1]]$retweetCount

# get favourite count
tweets[[1]]$favoriteCount
