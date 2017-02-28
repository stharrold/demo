## loading package dependencies
library(arules)
library(arulesViz)

## loading dataset
data(Groceries)

## exploring the data
inspect(Groceries[1:3])
# viewing the top ten purchased products
sort(itemFrequency(Groceries, type="absolute"), decreasing = TRUE)[1:10]
# visualizing the top ten purchased products
itemFrequencyPlot(Groceries,topN=10,type="absolute")

## association rule mining

# normal workflow
metric.params <- list(supp=0.001, conf=0.5)
rules <- apriori(Groceries, parameter = metric.params)
inspect(rules[1:5])

# pruning duplicate rules
prune.dup.rules <- function(rules){
  rule.subset.matrix <- is.subset(rules, rules)
  rule.subset.matrix[lower.tri(rule.subset.matrix, diag=T)] <- NA
  dup.rules <- colSums(rule.subset.matrix, na.rm=T) >= 1
  pruned.rules <- rules[!dup.rules]
  return(pruned.rules)
}

# sorting rules based on metrics
rules <- sort(rules, by="confidence", decreasing=TRUE)
rules <- prune.dup.rules(rules)
inspect(rules[1:5])

rules<-sort(rules, by="lift", decreasing=TRUE)
rules <- prune.dup.rules(rules)
inspect(rules[1:5])

## detecting specific item shopping patterns

# finding itemsets which lead to buying of an item on RHS
metric.params <- list(supp=0.001,conf = 0.5, minlen=2)
rules<-apriori(data=Groceries, parameter=metric.params, 
               appearance = list(default="lhs",rhs="soda"),
               control = list(verbose=F))
rules <- prune.dup.rules(rules)
rules<-sort(rules, decreasing=TRUE,by="confidence")
inspect(rules[1:5])

# finding items which are bought when we have an itemset on LHS
metric.params <- list(supp=0.001,conf = 0.3, minlen=2)
rules<-apriori(data=Groceries, parameter=metric.params, 
               appearance = list(default="rhs",lhs=c("yogurt", "sugar")),
               control = list(verbose=F))
#rules <- prune.dup.rules(rules)
rules<-sort(rules, decreasing=TRUE,by="confidence")
inspect(rules[1:5])

## visualizing rules
plot(rules,method="graph",interactive=TRUE,shading=TRUE)

