## The basics of using R

# TIP - You can select and execute the lines you want to or you can execute
# the whole file and see the different outputs. Recommended way is to 
# execute it section-wise.


## Using R as a scientific calculator

5 + 6

3 * 2

1 / 0


num <- 6
num ^ 2
num # a variable changes value only on re-assignment

num <- num ^ 2 * 5 + 10 / 3
num


## Operating on vectors

x <- 1:5
x

y <- c(6, 7, 8 ,9, 10)
y

z <- x + y
z


c(1,3,5,7,9) * 2
c(1,3,5,7,9) * c(2, 4) # here the second vector gets recycled

factorial(1:5)
exp(2:10)  # exponential function
cos(c(0, pi/4))  # cosine function
sqrt(c(1, 4, 9, 16))
sum(1:10)


## Special Values

1 / 0

0 / 0

Inf / NaN

Inf / Inf

log(Inf)

Inf + NA


vec <- c(0, Inf, NaN, NA)
is.finite(vec)
is.nan(vec)
is.na(vec)
is.infinite(vec)



