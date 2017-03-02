library(caret)  # feature selection algorithm
library(randomForest) # random forest algorithm

# rfe based feature selection algorithm
run.feature.selection <- function(num.iters=20, feature.vars, class.var){
  set.seed(10)
  variable.sizes <- 1:10
  control <- rfeControl(functions = rfFuncs, method = "cv", 
                        verbose = FALSE, returnResamp = "all", 
                        number = num.iters)
  results.rfe <- rfe(x = feature.vars, y = class.var, 
             sizes = variable.sizes, 
             rfeControl = control)
  return(results.rfe)
}

# run feature selection
rfe.results <- run.feature.selection(feature.vars=train.data[,-1], 
                                     class.var=train.data[,1])
# view results
rfe.results
