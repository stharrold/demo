library(caret) # nn models
library(ROCR) # evaluate models
source("performance_plot_utils.R") # plot curves
# data transformation
test.feature.vars <- test.data[,-1]
test.class.var <- test.data[,1]

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

## build model with training data
formula.init <- "credit.rating ~ ."
formula.init <- as.formula(formula.init)
nn.model <- train(formula.init, data = transformed.train, method="nnet")

## view model details
print(nn.model)

## predict and evaluate results
nn.predictions <- predict(nn.model, transformed.test.feature.vars, type="raw")
confusionMatrix(data=nn.predictions, reference=transformed.test.class.var, 
                positive="X1")


## nn specific feature selection
formula.init <- "credit.rating ~ ."
formula.init <- as.formula(formula.init)
control <- trainControl(method="repeatedcv", number=10, repeats=2)
model <- train(formula.init, data=transformed.train, method="nnet", 
               trControl=control)
importance <- varImp(model, scale=FALSE)
plot(importance, cex.lab=0.5)


## build new model with selected features
formula.new <- "credit.rating ~ account.balance + credit.purpose + savings + current.assets +
foreign.worker + previous.credit.payment.status"
formula.new <- as.formula(formula.new)
nn.model.new <- train(formula.new, data=transformed.train, method="nnet")

## predict and evaluate results
nn.predictions.new <- predict(nn.model.new, transformed.test.feature.vars, type="raw")
confusionMatrix(data=nn.predictions.new, reference=transformed.test.class.var, 
                positive="X1")

## view hyperparameter optimizations
plot(nn.model.new, cex.lab=0.5)


## plot model evaluation metric curves
nn.model.best <- nn.model
nn.predictions.best <- predict(nn.model.best, transformed.test.feature.vars, type="prob")
nn.prediction.values <- nn.predictions.best[,2]
predictions <- prediction(nn.prediction.values, test.class.var)
par(mfrow=c(1,2))
plot.roc.curve(predictions, title.text="NN ROC Curve")
plot.pr.curve(predictions, title.text="NN Precision/Recall Curve")