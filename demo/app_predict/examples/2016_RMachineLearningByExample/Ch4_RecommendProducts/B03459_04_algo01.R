############################################################################
# Chapter      :   4 
# Algorithm    :   Matrix Factorization based User Collaborative Filtering
# Description  :   The following code predicts product ratings for
#				           active user using matrix factorization.
############################################################################



##################################
# Matrix Factorization algorithm
##################################
mf_based_ucf <- function(ratings_matrix, X, Y, K, epoch=5000, alpha=0.0002, beta=0.02){
  
  #transpose Y
  Y <- t(Y)
  
  # Iterate epoch number of times
  for (step in seq(epoch)){
    for (i in seq(nrow(ratings_matrix))){
      for (j in seq(length(ratings_matrix[i, ]))){
        if (ratings_matrix[i, j] > 0){
          # error 
          eij = ratings_matrix[i, j] - as.numeric(X[i, ] %*% Y[, j])
		      
          # gradient calculation 
          for (k in seq(K)){
            X[i, k] = X[i, k] + alpha * (2 * eij * Y[k, j] - beta * X[i, k])
            Y[k, j] = Y[k, j] + alpha * (2 * eij * X[i, k] - beta * Y[k, j])
          }
        }
      }
    }
    
    # Overall Squared Error Calculation
    e = 0
    
    for (i in seq(nrow(ratings_matrix))){
      for (j in seq(length(ratings_matrix[i, ]))){
        if (ratings_matrix[i, j] > 0){
          e = e + (ratings_matrix[i, j] - as.numeric(X[i, ] %*% Y[, j]))^2
          for (k in seq(K)){
            e = e + (beta/2) * (X[i, k]^2 + Y[k, j]^2)
          }
        }
      }
    }
    
    # stop if error falls below this threshold
    if (e < 0.001){
      break
    }
  }
  
  #inner product
  pR <- X %*% Y
  pR <- round(pR, 2)
  return (pR)
}


##################################
# Setup Constants and Data
##################################


# load raw ratings from csv
raw_ratings <- read.csv("product_ratings.csv")

# convert columnar data to sparse ratings matrix
ratings_matrix <- data.matrix(raw_ratings)

# number of rows in ratings
rows <- nrow(ratings_matrix)

# number of columns in ratings matrix
columns <- ncol(ratings_matrix)

# latent features
K <- 2

# User-Feature Matrix
X <- matrix(runif(rows*K), nrow=rows, byrow=TRUE)

# Item-Feature Matrix
Y <- matrix(runif(columns*K), nrow=columns, byrow=TRUE)

# iterations
epoch <- 10000

# rate of descent
alpha <- 0.0002

# regularization constant
beta <- 0.02


pred.matrix <- mf_based_ucf(ratings_matrix, X, Y, K, epoch = epoch)

# setting column names
colnames(pred.matrix)<-c("iPhone.4","iPhone.5s","Nexus.5","Moto.X","Moto.G","Nexus.6","One.Plus.One")

pred.matrix

