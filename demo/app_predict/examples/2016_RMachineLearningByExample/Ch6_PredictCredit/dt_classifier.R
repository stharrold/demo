library(rpart)# tree models 
library(caret) # feature selection
library(rpart.plot) # plot dtree
library(ROCR) # model evaluation
library(e1071) # tuning model
source("performance_plot_utils.R") # plotting curves

## separate feature and class variables
test.feature.vars <- test.data[,-1]
test.class.var <- test.data[,1]

## build initial model with training data
formula.init <- "credit.rating ~ ."
formula.init <- as.formula(formula.init)
dt.model <- rpart(formula=formula.init, method="class",data=train.data, 
                  control = rpart.control(minsplit=20, cp=0.05))

## predict and evaluate results
dt.predictions <- predict(dt.model, test.feature.vars, type="class")
confusionMatrix(data=dt.predictions, reference=test.class.var, positive="1")


## dt specific feature selection
formula.init <- "credit.rating ~ ."
formula.init <- as.formula(formula.init)
control <- trainControl(method="repeatedcv", number=10, repeats=2)
model <- train(formula.init, data=train.data, method="rpart", 
               trControl=control)
importance <- varImp(model, scale=FALSE)
plot(importance, cex.lab=0.5)


## build new model with selected features
formula.new <- "credit.rating ~ account.balance + savings +
                                credit.amount + credit.duration.months + 
                                previous.credit.payment.status"
formula.new <- as.formula(formula.new)
dt.model.new <- rpart(formula=formula.new, method="class",data=train.data, 
                  control = rpart.control(minsplit=20, cp=0.05),
                  parms = list(prior = c(0.7, 0.3)))

## predict and evaluate results
dt.predictions.new <- predict(dt.model.new, test.feature.vars, type="class")
confusionMatrix(data=dt.predictions.new, reference=test.class.var, positive="1")


# view model details
dt.model.best <- dt.model.new
print(dt.model.best)
par(mfrow=c(1,1))
prp(dt.model.best, type=1, extra=3, varlen=0, faclen=0)


## plot model evaluation metric curves
dt.predictions.best <- predict(dt.model.best, test.feature.vars, type="prob")
dt.prediction.values <- dt.predictions.best[,2]
predictions <- prediction(dt.prediction.values, test.class.var)
par(mfrow=c(1,2))
plot.roc.curve(predictions, title.text="DT ROC Curve")
plot.pr.curve(predictions, title.text="DT Precision/Recall Curve")