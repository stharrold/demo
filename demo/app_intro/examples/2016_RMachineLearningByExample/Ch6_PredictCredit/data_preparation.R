# load the dataset into data frame
credit.df <- read.csv("credit_dataset_final.csv", header = TRUE, sep = ",")

## data type transformations - factoring
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

## normalizing - scaling
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center=T, scale=T)
  }
  return(df)
}

# normalize variables
numeric.vars <- c("credit.duration.months", "age", "credit.amount")
credit.df <- scale.features(credit.df, numeric.vars)
# factor variables
categorical.vars <- c('credit.rating', 'account.balance', 'previous.credit.payment.status',
                      'credit.purpose', 'savings', 'employment.duration', 'installment.rate',
                      'marital.status', 'guarantor', 'residence.duration', 'current.assets',
                      'other.credits', 'apartment.type', 'bank.credits', 'occupation', 
                      'dependents', 'telephone', 'foreign.worker')
credit.df <- to.factors(df=credit.df, variables=categorical.vars)

# split data into training and test datasets in 60:40 ratio
indexes <- sample(1:nrow(credit.df), size=0.6*nrow(credit.df))
train.data <- credit.df[indexes,]
test.data <- credit.df[-indexes,]