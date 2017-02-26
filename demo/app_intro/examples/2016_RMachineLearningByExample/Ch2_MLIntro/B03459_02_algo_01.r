###############################################################
#Chapter      :   2 
#Algorithm    :   Perceptron
# Description :   The following code models a perceptron and 
#                 shows its progress after every iteration.
###############################################################


x1 <- runif(30,-1,1) #30 random numbers between -1 and 1 which are uniformly distributed
x2 <- runif(30,-1,1)

#form the input vector x
x <- cbind(x1,x2) 

#generate output vector
Y <- ifelse(x2>0.5+x1,+1,-1) 

#plot the actual separator
plot(x,pch=ifelse(Y>0,"+","-"),xlim=c(-1,1),ylim=c(-1,1),cex=2) #cex is zoom
abline(0.5,1) #correct Y

#function to calculate distance from hyper plane
calculate_distance = function(x,w,b) {
sum(x*w) + b
}

#linear classifier
linear_classifier = function(x,w,b) {
distances =apply(x, 1, calculate_distance, w, b)
return(ifelse(distances < 0, -1, +1))
}


#test the classifier
linear_classifier(x,c(-1,1)/sqrt(2),-sqrt(2)/4)


#function to calculate 2nd norm
second_norm = function(x) {sqrt(sum(x * x))}

#perceptron training algorithm
perceptron = function(x, y, learning_rate=1) {

w = vector(length = ncol(x)) # initialize w
b = 0 # Initialize b
k = 0 # count iterations

R = max(apply(x, 1, second_norm))#constant with value greater than distance of furthest point

incorrect = TRUE # flag to identify classifier
#initialize plot
plot(x,cex=0.2)

#loop till correct classifier is not found
while (incorrect ) {

incorrect = FALSE
#classify with current weights
yc <- linear_classifier(x,w,b)
#Loop over each point in the input x
for (i in 1:nrow(x)) {
#update weights if point not classified correctly
if (y[i] != yc[i]) {
w <- w + learning_rate * y[i]*x[i,]
b <- b + learning_rate * y[i]*R^2
k <- k+1

#currect classifier's components
if(k%%5 == 0){
	intercept <- - b / w[[2]]
	slope <- - w[[1]] / w[[2]]
	#plot the classifier hyper plane
	abline(intercept,slope,col="red")
	#wait for user input
	cat ("Iteration # ",k,"\n")
	cat ("Press [enter] to continue")
	line <- readline()
}
incorrect =TRUE
}
} }

s = second_norm(w)
#scale the classifier with unit vector
return(list(w=w/s,b=b/s,updates=k))
}

#train the perceptron
p <- perceptron(x,Y)

#classify based on calculated separator
y <- linear_classifier(x,p$w,p$b)


plot(x,cex=0.2)
#zoom into points near the separator and color code them
points(subset(x,Y==1),col="black",pch="+",cex=2)
points(subset(x,Y==-1),col="red",pch="-",cex=2)

# compute intercept on y axis of separator
# from w and b
intercept <- - p$b / p$w[[2]]
# compute slope of separator from w
slope <- - p$w[[1]] /p$ w[[2]]
# draw separating boundary
abline(intercept,slope,col="green")