source('descriptive_analytics_utils.R')

## Getting the Data

# load in the data and attach the data frame
credit.df <- read.csv("german_credit_dataset.csv", header = TRUE, sep = ",")

# class should be data.frame
class(credit.df)

# get a quick peek at the data
head(credit.df)

# get dataset detailed info
str(credit.df)



## Data pre-processing

# check if data frame contains NA values
sum(is.na(credit.df))

# check if total records reduced after removing rows with NA values
sum(complete.cases(credit.df))

# select variables for data transformation
categorical.vars <- c('credit.rating', 'account.balance', 'previous.credit.payment.status',
                      'credit.purpose', 'savings', 'employment.duration', 'installment.rate',
                      'marital.status', 'guarantor', 'residence.duration', 'current.assets',
                      'other.credits', 'apartment.type', 'bank.credits', 'occupation',
                      'dependents', 'telephone', 'foreign.worker')

# transform data types
credit.df <- to.factors(df = credit.df, variables = categorical.vars)

# verify transformation in data frame details
str(credit.df)



## Data analysis

#load dependencies
library(car)
# access dataset features directly
attach(credit.df)


# credit.rating stats
get.categorical.variable.stats(credit.rating)

# credit.rating visualizations
visualize.barchart(credit.rating)


# account.balance stats and bar chart
get.categorical.variable.stats(account.balance)
visualize.barchart(account.balance)

# recode classes and update data frame
new.account.balance <- recode(account.balance,
                          "1=1;2=2;3=3;4=3")
credit.df$account.balance <- new.account.balance

# contingency table and mosaic plot
get.contingency.table(credit.rating, new.account.balance)
visualize.contingency.table(credit.rating, new.account.balance)


# credit.duration.months analysis
get.numeric.variable.stats(credit.duration.months)

# histogram\density plot
visualize.distribution(credit.duration.months)

# box plot
visualize.boxplot(credit.duration.months, credit.rating)


# previous.credit.payment.status stats and bar chart
get.categorical.variable.stats(previous.credit.payment.status)
visualize.barchart(previous.credit.payment.status)

# recode classes and update data frame
new.previous.credit.payment.status <- recode(previous.credit.payment.status,
                                             "0=1;1=1;2=2;3=3;4=3")
credit.df$previous.credit.payment.status <- new.previous.credit.payment.status

# contingency table
get.contingency.table(credit.rating, new.previous.credit.payment.status)


# credit.purpose stats and bar chart
get.categorical.variable.stats(credit.purpose)
visualize.barchart(credit.purpose)

# recode classes and update data frame
new.credit.purpose <- recode(credit.purpose,"0=4;1=1;2=2;3=3;
                                             4=3;5=3;6=3;7=4;
                                             8=4;9=4;10=4")
credit.df$credit.purpose <- new.credit.purpose

# contingency table
get.contingency.table(credit.rating, new.credit.purpose)


# credit.amount analysis
get.numeric.variable.stats(credit.amount)

# histogram\density plot
visualize.distribution(credit.amount)

# box plot
visualize.boxplot(credit.amount, credit.rating)


# feature: savings - recode classes and update data frame
new.savings <- recode(savings,"1=1;2=2;3=3;
                               4=3;5=4")
credit.df$savings <- new.savings

# contingency table
get.contingency.table(credit.rating, new.savings)


# feature: employment.duration - recode classes and update data frame
new.employment.duration <- recode(employment.duration,
                                  "1=1;2=1;3=2;4=3;5=4")
credit.df$employment.duration <- new.employment.duration

# contingency table
get.contingency.table(credit.rating, new.employment.duration)


# feature: installment.rate - contingency table and statistical tests
get.contingency.table(credit.rating, installment.rate,
                     stat.tests=TRUE)


# feature: marital.status - recode classes and update data frame
new.marital.status <- recode(marital.status, "1=1;2=1;3=3;4=4")
credit.df$marital.status <- new.marital.status

# contingency table
get.contingency.table(credit.rating, new.marital.status)


# feature: guarantor - recode classes and update data frame
new.guarantor <- recode(guarantor, "1=1;2=2;3=2")
credit.df$guarantor <- new.guarantor

# perform statistical tests
fisher.test(credit.rating, new.guarantor)
chisq.test(credit.rating, new.guarantor)


# perform statistical tests for residence.duration
fisher.test(credit.rating, residence.duration)
chisq.test(credit.rating, residence.duration)


# perform statistical tests for current.assets
fisher.test(credit.rating, current.assets)
chisq.test(credit.rating, current.assets)


# age analysis
get.numeric.variable.stats(age)

# histogram\density plot
visualize.distribution(age)

# box plot
visualize.boxplot(age, credit.rating)


# feature: other.credits - recode classes and update data frame
new.other.credits <- recode(other.credits, "1=1;2=1;3=2")
credit.df$other.credits <- new.other.credits

# perform statistical tests
fisher.test(credit.rating, new.other.credits)
chisq.test(credit.rating, new.other.credits)


# perform statistical tests for apartment.type
fisher.test(credit.rating, apartment.type)
chisq.test(credit.rating, apartment.type)


# feature: bank.credits - recode classes and update data frame
new.bank.credits <- recode(bank.credits, "1=1;2=2;3=2;4=2")
credit.df$bank.credits <- new.bank.credits

# perform statistical tests
fisher.test(credit.rating, new.bank.credits)
chisq.test(credit.rating, new.bank.credits)


# perform statistical tests for occupation
fisher.test(credit.rating, occupation)
chisq.test(credit.rating, occupation)


# perform statistical tests for dependents
fisher.test(credit.rating, dependents)
chisq.test(credit.rating, dependents)


# perform statistical tests for telephone
fisher.test(credit.rating, telephone)
chisq.test(credit.rating, telephone)


# perform statistical tests for foreign.worker
fisher.test(credit.rating, foreign.worker)
chisq.test(credit.rating, foreign.worker)


## Save the transformed dataset
write.csv(file='credit_dataset_final.csv', x = credit.df,
          row.names = F)
