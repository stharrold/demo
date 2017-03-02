
# loading the required package
library(arules)

# getting and loading the data
data(Groceries)

# inspecting the first 3 transactions 
inspect(Groceries[1:3])


# count based product contigency matrix 
ct <- crossTable(Groceries, measure="count", sort=TRUE)
ct[1:5, 1:5]

# support based product contigency matrix 
ct <- crossTable(Groceries, measure="support", sort=TRUE)
ct[1:5, 1:5]

# lift based product contigency matrix 
ct <- crossTable(Groceries, measure="lift", sort=TRUE)
ct[1:5, 1:5]

