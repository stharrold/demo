## Working with functions

# TIP - You can select and execute the lines you want to or you can execute
# the whole file and see the different outputs. Recommended way is to 
# execute it section-wise.


## Built-in Functions

sqrt(5)
sqrt(c(1,2,3,4,5,6,7,8,9,10))

# aggregating functions
mean(c(1,2,3,4,5,6,7,8,9,10))
median(c(1,2,3,4,5,6,7,8,9,10))


## User-defined Functions

square <- function(data){
  return (data^2)
}

square(5)
square(c(1,2,3,4,5))

point <- function(xval, yval){
  return (c(x=xval,y=yval))
}

p1 <- point(5,6)
p2 <- point(2,3)

p1
p2


## Passing Functions as Arguments

# defining the function
euclidean.distance <- function(point1, point2, square.func){
  distance <- sqrt(
                  as.integer(
                    square.func(point1['x'] - point2['x'])
                  ) +
                  as.integer(
                    square.func(point1['y'] - point2['y'])
                  )
              )
  return (c(distance=distance))
}

# executing the function, passing square as argument
euclidean.distance(point1 = p1, point2 = p2, square.func = square)
euclidean.distance(point1 = p2, point2 = p1, square.func = square)
euclidean.distance(point1 = point(10, 3), point2 = point(-4, 8), square.func = square)

