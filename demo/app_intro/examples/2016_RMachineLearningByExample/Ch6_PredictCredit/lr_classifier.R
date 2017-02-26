library(caret) # model training and evaluation
library(ROCR) # model evaluation
source("performance_plot_utils.R") # plotting metric results

## separate feature and class variables
test.feature.vars <- test.data[,-1]
test.class.var <- test.data[,1]

# build a logistic regression model
formula.init <- "credit.rating ~ ."
formula.init <- as.formula(formula.init)
lr.model <- glm(formula=formula.init, data=train.data, family="binomial")

# view model details
summary(lr.model)

# perform and evaluate predictions
lr.predictions <- predict(lr.model, test.data, type="response")
lr.predictions <- round(lr.predictions)
confusionMatrix(data=lr.predictions, reference=test.class.var, positive='1')


## glm specific feature selection
formula <- "credit.rating ~ ."
formula <- as.formula(formula)
control <- trainControl(method="repeatedcv", number=10, repeats=2)
model <- train(formula, data=train.data, method="glm", 
               trControl=control)
importance <- varImp(model, scale=FALSE)
plot(importance)


# build new model with selected features
formula.new <- "credit.rating ~ account.balance + credit.purpose + previous.credit.payment.status 
                                + savings + credit.duration.months"
formula.new <- as.formula(formula.new)
lr.model.new <- glm(formula=formula.new, data=train.data, family="binomial")

# view model details
summary(lr.model.new)

# perform and evaluate predictions 
lr.predictions.new <- predict(lr.model.new, test.data, type="response") 
lr.predictions.new <- round(lr.predictions.new)
confusionMatrix(data=lr.predictions.new, reference=test.class.var, positive='1')



## model performance evaluations

# plot best model evaluation metric curves
lr.model.best <- lr.model
lr.prediction.values <- predict(lr.model.best, test.feature.vars, type="response")
predictions <- prediction(lr.prediction.values, test.class.var)
par(mfrow=c(1,2))
plot.roc.curve(predictions, title.text="LR ROC Curve")
plot.pr.curve(predictions, title.text="LR Precision/Recall Curve")