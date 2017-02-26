###############################################################
#Chapter      :   2 
#Algorithm    :   Linear Regression
# Description :   The following code fits a linear regression 
#                 model for a height weight sample of 19 students
###############################################################

#Height and weight vectors for 19 children

height <- c(69.1,56.4,65.3,62.8,63,57.3,59.8,62.5,62.5,59.0,51.3,64,56.4,66.5,72.2,65.0,67.0,57.6,66.6)

weight <- c(113,84,99,103,102,83,85,113,84,99,51,90,77,112,150,128,133,85,112)

plot(height,weight)
cor(height,weight)

#Fitting the linear model

model <- lm(weight ~ height) # weight = slope*weight + intercept

#get the intercept(b0) and the slope(b1) values
model

#check all attributes calculated by lm
attributes(model)

#getting only the intercept
model$coefficients[1] #or model$coefficients[[1]]

#getting only the slope
model$coefficients[2] #or model$coefficients[[2]]

#checking the residuals
residuals(model)

#predicting the weight for a given height, say 60 inches
model$coefficients[[2]]*50 + model$coefficients[[1]]

#detailed information about the model
summary(model)

#plot data points
plot(height,weight)

#draw the regression line
abline(model)

