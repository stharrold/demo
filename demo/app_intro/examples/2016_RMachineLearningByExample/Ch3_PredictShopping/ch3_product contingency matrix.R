
## Getting the data

# reading in the dataset
data <- read.csv("top_supermarket_transactions.csv")

# assigning row names to be same as column names 
# to build the contingency matrix
row.names(data) <- data[[1]]
data <- subset(data, select = c(-1))


## viewing the contingency matrix
cat("Products Transactions Contingency Matrix")
data 


## Analyzing and visualizing the data

# Frequency of products bought with milk
data['milk', ]

# Sorting to get top products bought with milk
sort(data['milk', ], decreasing = TRUE)

# Frequency of products bought with bread
data['bread', ]

# Sorting to get top products bought with bread
sort(data['bread', ], decreasing = TRUE)

# Visualizing the data
mosaicplot(as.matrix(data), 
           color=TRUE, 
           title(main="Products Contingency Mosaic Plot"),
           las=2
           )

## Global Recommendations
cat("Recommendatons based on global products contingency matrix")
items <- names(data)
for (item in items){
  cat(paste("Top 2 recommended items to buy with", item, "are: "))
  item.data <- subset(data[item,], select=names(data)[!names(data) %in% item])
  cat(names(item.data[order(item.data, decreasing = TRUE)][c(1,2)]))
  cat("\n")
}
