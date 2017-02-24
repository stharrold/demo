library(e1071) # svm model
library(caret) # model training\optimizations
library(kernlab) # svm model for hyperparameters
library(ROCR) # model evaluation
source("performance_plot_utils.R") # plot model metrics


## separate feature and class variables
test.feature.vars <- test.data[,-1]
test.class.var <- test.data[,1]

## build initial model with training data
formula.init <- "credit.rating ~ ."
formula.init <- as.formula(formula.init)
svm.model <- svm(formula=formula.init, data=train.data, 
                 kernel="radial", cost=100, gamma=1)

## view inital model details
summary(svm.model)

## predict and evaluate results
svm.predictions <- predict(svm.model, test.feature.vars)
confusionMatrix(data=svm.predictions, reference=test.class.var, positive="1")


## svm specific feature selection
formula.init <- "credit.rating ~ ."
formula.init <- as.formula(formula.init)
control <- trainControl(method="repeatedcv", number=10, repeats=2)
model <- train(formula.init, data=train.data, method="svmRadial", 
               trControl=control)
importance <- varImp(model, scale=FALSE)
plot(importance, cex.lab=0.5)


## build new model with selected features
formula.new <- "credit.rating ~ account.balance + credit.duration.months +
                            savings + previous.credit.payment.status +
                            credit.amount"
formula.new <- as.formula(formula.new)
svm.model.new <- svm(formula=formula.new, data=train.data, 
                 kernel="radial", cost=10, gamma=0.25)

## predict results with new model on test data
svm.predictions.new <- predict(svm.model.new, test.feature.vars)

## new model performance evaluation
confusionMatrix(data=svm.predictions.new, reference=test.class.var, positive="1")



## hyperparameter optimizations

# run grid search
cost.weights <- c(0.1, 10, 100)
gamma.weights <- c(0.01, 0.25, 0.5, 1)
tuning.results <- tune(svm, formula.new, 
                       data = train.data, kernel="Radial", 
                       ranges=list(cost=cost.weights, gamma=gamma.weights))

# view optimization results
print(tuning.results)

# plot results
plot(tuning.results, cex.main=0.6, cex.lab=0.8,xaxs="i", yaxs="i")

# get best model and evaluate predictions
svm.model.best = tuning.results$best.model
svm.predictions.best <- predict(svm.model.best, test.feature.vars)
confusionMatrix(data=svm.predictions.best, reference=test.class.var, positive="1")


# plot best model evaluation metric curves
svm.predictions.best <- predict(svm.model.best, test.feature.vars, decision.values = T)
svm.prediction.values <- attributes(svm.predictions.best)$decision.values
predictions <- prediction(svm.prediction.values, test.class.var)
par(mfrow=c(1,2))
plot.roc.curve(predictions, title.text="SVM ROC Curve")
plot.pr.curve(predictions, title.text="SVM Precision/Recall Curve")



## model optimizations based on ROC

# data transformation
transformed.train <- train.data
transformed.test <- test.data
for (variable in categorical.vars){
  new.train.var <- make.names(train.data[[variable]])
  transformed.train[[variable]] <- new.train.var
  new.test.var <- make.names(test.data[[variable]])
  transformed.test[[variable]] <- new.test.var
}
transformed.train <- to.factors(df=transformed.train, variables=categorical.vars)
transformed.test <- to.factors(df=transformed.test, variables=categorical.vars)
transformed.test.feature.vars <- transformed.test[,-1]
transformed.test.class.var <- transformed.test[,1]

# view data to understand transformations
summary(transformed.train$credit.rating)

# build optimal model based on AUC
grid <- expand.grid(C=c(1,10,100), 
                    sigma=c(0.01, 0.05, 0.1, 0.5, 1))
ctr <- trainControl(method='cv', number=10,
                    classProbs=TRUE,
                    summaryFunction=twoClassSummary)
svm.roc.model <- train(formula.init, transformed.train,
                       method='svmRadial', trControl=ctr, 
                       tuneGrid=grid, metric="ROC")

# predict and evaluate model performance
predictions <- predict(svm.roc.model, transformed.test.feature.vars)
confusionMatrix(predictions, transformed.test.class.var, positive = "X1")


## plot model evaluation metric curves
svm.predictions <- predict(svm.roc.model, transformed.test.feature.vars, type="prob")
svm.prediction.values <- svm.predictions[,2]
predictions <- prediction(svm.prediction.values, test.class.var)
par(mfrow=c(1,2))
plot.roc.curve(predictions, title.text="SVM ROC Curve")
plot.pr.curve(predictions, title.text="SVM Precision/Recall Curve")