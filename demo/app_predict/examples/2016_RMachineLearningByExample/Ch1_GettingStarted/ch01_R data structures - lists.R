## Data Structures in R - Lists

# TIP - You can select and execute the lines you want to or you can execute
# the whole file and see the different outputs. Recommended way is to 
# execute it section-wise.


## Creating and Indexing Lists

list.sample <- list(
  1:5,
  c("first", "second", "third"),
  c(TRUE, FALSE, TRUE, TRUE),
  cos,
  matrix(1:9, nrow = 3, ncol = 3)
)
list.sample

list.with.names <- list(
  even.nums = seq.int(2,10,2),
  odd.nums  = seq.int(1,10,2),
  languages = c("R", "Python", "Julia", "Java"),
  cosine.func = cos
)
list.with.names

list.with.names$cosine.func
list.with.names$cosine.func(pi)
list.sample[[4]]
list.sample[[4]](pi)
list.with.names$odd.nums
list.sample[[1]]
list.sample[[3]]


## Combining and Converting Lists

l1 <- list(
  nums = 1:5,
  chars = c("a", "b", "c", "d", "e"),
  cosine = cos
)

l2 <- list(
  languages = c("R", "Python", "Java"),
  months = c("Jan", "Feb", "Mar", "Apr"),
  sine = sin
)

l3 <- c(l1, l2)
l3

l1 <- 1:5
class(l1)
list.l1 <- as.list(l1)
class(list.l1)
list.l1
unlist(list.l1)



