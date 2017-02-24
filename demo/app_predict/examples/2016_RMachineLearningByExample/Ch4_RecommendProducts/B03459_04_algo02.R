############################################################################
# Chapter      :   4 
# Algorithm    :   Recommenderlab based User Collaborative Filtering
# Description  :   The following code utilizes recommenderlab to predict 
#                  product ratings and evaluate the model
############################################################################

# Load recommenderlab library
library("recommenderlab")

# Read dataset from csv file
raw_data <- read.csv("product_ratings_data.csv")

# Create rating matrix from data 
ratings_matrix<- as(raw_data, "realRatingMatrix")

############### Exploring Data ##################

#view transformed data
image(ratings_matrix[1:6,1:10])

# Extract a sample from ratings matrix
sample_ratings <-sample(ratings_matrix,1000)

# Get the mean product ratings as given by first user
rowMeans(sample_ratings[1,])

# Get distribution of item ratings
hist(getRatings(sample_ratings), breaks=100,xlab = "Product Ratings",main = " Histogram of Product Ratings")

# Get distribution of normalized item ratings
hist(getRatings(normalize(sample_ratings)),breaks=100, xlab = "Normalized Product Ratings",main = " Histogram of Normalized Product Ratings")

# Number of items rated per user
hist(rowCounts(sample_ratings),breaks=50,xlab = "Number of Products",main = " Histogram of Rated Products Distribution")


############### Prepare Recommendation Model and Predict ##################

# Create 'User Based collaborative filtering' model 
ubcf_recommender <- Recommender(ratings_matrix[1:1000],"UBCF")

# Predict list of product which can be recommended to given users
recommendations <- predict(ubcf_recommender, ratings_matrix[1010:1011], n=5)

# show recommendation in form of the list
as(recommendations, "list")


############### Testing of the recommender algorithm ##################

# Evaluation scheme
eval_scheme <- evaluationScheme(ratings_matrix[1:500],method="split",train=0.9,given=15)

# View the evaluation scheme
eval_scheme

# Training model using UBCF
training_recommender <- Recommender(getData(eval_scheme, "train"), "UBCF")

# Preditions on the test dataset
test_rating <- predict(training_recommender, getData(eval_scheme, "known"), type="ratings")

#Error 
error <- calcPredictionAccuracy(test_rating, getData(eval_scheme, "unknown"))

error

# Training model using IBCF
training_recommender_2 <- Recommender(getData(eval_scheme, "train"), "IBCF")

# Preditions on the test dataset
test_rating_2 <- predict(training_recommender_2, getData(eval_scheme, "known"), type="ratings")

error_compare <- rbind(calcPredictionAccuracy(test_rating, getData(eval_scheme, "unknown")),
                       calcPredictionAccuracy(test_rating_2, getData(eval_scheme, "unknown")))

rownames(error_compare) <- c("User Based CF","Item Based CF")

error_compare

## evaluate topNLists instead (you need to specify given and goodRating!)
test_rating_3 <- predict(training_recommender, getData(eval_scheme, "known"), type="topNList")

# Calculate Prediction Accuracy
calcPredictionAccuracy(test_rating_3, getData(eval_scheme, "unknown"), given=15,goodRating=5) 




