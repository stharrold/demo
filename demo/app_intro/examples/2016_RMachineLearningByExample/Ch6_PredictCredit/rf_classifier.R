library(randomForest) #rf model
library(caret) # feature selection
library(e1071) # model tuning
library(ROCR) # model evaluation
source("performance_plot_utils.R") # plot curves
## separate feature and class variables
test.feature.vars <- test.data[,-1]
test.class.var <- test.data[,1]

## build initial model with training data
formula.init <- "credit.rating ~ ."
formula.init <- as.formula(formula.init)
rf.model <- randomForest(formula.init, data = train.data, importance=T, proximity=T)

## view model details
print(rf.model)

## predict and evaluate results
rf.predictions <- predict(rf.model, test.feature.vars, type="class")
confusionMatrix(data=rf.predictions, reference=test.class.var, positive="1")


## build new model with selected features
formula.new <- "credit.rating ~ account.balance + savings +
                                credit.amount + credit.duration.months + 
                                previous.credit.payment.status"
formula.new <- as.formula(formula.new)
rf.model.new <- randomForest(formula.new, data = train.data, 
                         importance=T, proximity=T)

## predict and evaluate results
rf.predictions.new <- predict(rf.model.new, test.feature.vars, type="class")
confusionMatrix(data=rf.predictions.new, reference=test.class.var, positive="1")


## hyperparameter optimizations

# run grid search
nodesize.vals <- c(2, 3, 4, 5)
ntree.vals <- c(200, 500, 1000, 2000)
tuning.results <- tune.randomForest(formula.new, 
                             data = train.data,
                             mtry=3, 
                             nodesize=nodesize.vals,
                             ntree=ntree.vals)
print(tuning.results)

# get best model and predict and evaluate performance
rf.model.best <- tuning.results$best.model
rf.predictions.best <- predict(rf.model.best, test.feature.vars, type="class")
confusionMatrix(data=rf.predictions.best, reference=test.class.var, positive="1")


## plot model evaluation metric curves
rf.predictions.best <- predict(rf.model.best, test.feature.vars, type="prob")
rf.prediction.values <- rf.predictions.best[,2]
predictions <- prediction(rf.prediction.values, test.class.var)
par(mfrow=c(1,2))
plot.roc.curve(predictions, title.text="RF ROC Curve")
plot.pr.curve(predictions, title.text="RF Precision/Recall Curve")