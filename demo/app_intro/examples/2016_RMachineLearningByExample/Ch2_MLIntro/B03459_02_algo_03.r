###############################################################
#Chapter      :   2 
#Algorithm    :   KNN
# Description :   The following code classifies IRIS dataset
#                 using KNN algorithm
###############################################################

#this should print the contents of data set onto the console
iris

#skip these steps if you already have iris on your system
iris <- read.csv(url("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), header = FALSE)
#assign proper headers
names(iris) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")


#to view top few rows of data
head(iris)

#to view data types, sample values, categorical values, etc
str(iris)

#detailed view of the data set
summary(iris)


#to view top few rows of data
head(iris)

#to view data types, sample values, categorical values, etc
str(iris)

#detailed view of the data set
summary(iris)

#load the package
library(ggvis)

#plot the species
iris %>% ggvis(~Petal.Length, ~Petal.Width, shape = ~factor(Species)) %>% layer_points()

#normalization function

min_max_normalizer <- function(x) 
{ 
  num <- x - min(x) 
  denom <- max(x) - min(x) 
  return (num/denom) 
}

#normalizing iris data set
normalized_iris <- as.data.frame(lapply(iris[1:4], min_max_normalizer))


#viewing normalized data
summary(normalized_iris)

#checking the data constituency
table(iris$Species)

#set seed for randomization
set.seed(1234)

# setting the training-test split to 67% and 37% respectively
random_samples <- sample(2, nrow(iris), replace=TRUE, prob=c(0.67, 0.33))


# training data set
iris.training <- iris[
  random_samples ==1, 1:4] 

#training labels
iris.trainLabels <- iris[
  random_samples ==1, 5]


# test data set
iris.test <- iris[
  random_samples ==2, 1:4]

#testing labels
iris.testLabels <- iris[
  random_samples ==2, 5]

#setting library
library(class)

#executing knn for k=3
iris_model <- knn(train = iris.training, test = iris.test, cl = iris.trainLabels, k=3)

#summary of the model learnt
iris_model


#setting library
library(gmodels)

#Preparing cross table
CrossTable(x = iris.testLabels, y = iris_model, prop.chisq=FALSE)

