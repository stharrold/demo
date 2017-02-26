###############################################################
#Chapter      :   2 
#Algorithm    :   Apriori
# Description :   The following code uses transactions from Adult
#                 dataset to find association rules using Apriori
###############################################################

# setting the apriori library
library(arules)

# loading data
data("Adult");


# summary of data set
summary(Adult);

# Sample 5 records
inspect(Adult[0:5]);


# executing apriori with support=50% confidence =80%
rules <- apriori(Adult, parameter=list(support=0.5, confidence=0.8,target="rules"));

# view a summary
summary(rules);

#view top 3 rules
as(head(sort(rules, by = c("confidence", "support")), n=3), "data.frame")

