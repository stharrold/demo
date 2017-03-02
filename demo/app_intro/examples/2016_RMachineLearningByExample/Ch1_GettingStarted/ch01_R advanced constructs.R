## Advanced constructs

# TIP - You can select and execute the lines you want to or you can execute
# the whole file and see the different outputs. Recommended way is to 
# execute it section-wise.


## lapply and sapply

# lapply function definition
lapply

# example
nums <- list(l1=c(1,2,3,4,5,6,7,8,9,10), l2=1000:1020)
lapply(nums, mean)

data <- list(l1=1:10, l2=runif(10), l3=rnorm(10,2))
data

lapply(data, mean)
sapply(data, mean)


## apply

mat <- matrix(rnorm(20), nrow=5, ncol=4)
mat

# row sums
apply(mat, 1, sum)
rowSums(mat)

# row means
apply(mat, 1, mean)
rowMeans(mat)

# col sums
apply(mat, 2, sum)
colSums(mat)

# col means
apply(mat, 2, mean)
colMeans(mat)

# row quantiles
apply(mat, 1, quantile, probs=c(0.25, 0.5, 0.75))


## tapply

data <- c(1:10, rnorm(10,2), runif(10))
data

groups <- gl(3,10)
groups

tapply(data, groups, mean)
tapply(data, groups, mean, simplify = FALSE)
tapply(data, groups, range)


## mapply

list(rep(1,4), rep(2,3), rep(3,2), rep(4,1))
mapply(rep, 1:4, 4:1)


