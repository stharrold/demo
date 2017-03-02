## Data Structures in R - Vectors

# TIP - You can select and execute the lines you want to or you can execute
# the whole file and see the different outputs. Recommended way is to 
# execute it section-wise.


## Creating vectors

c(2.5:4.5, 6, 7, c(8, 9, 10), c(12:15))

vector("numeric", 5)

vector("logical", 5)

logical(5)

# seq is a function which creates sequences
seq.int(1,10)

seq.int(1,10,2)

seq_len(10)


## Indexing and Naming Vectors

vec <- c("R", "Python", "Julia", "Haskell", "Java", "Scala")
vec[1]
vec[2:4]
vec[c(1, 3, 5)]

nums <- c(5, 8, 10, NA, 3, 11)
nums
which.min(nums)   # index of the minimum element
which.max(nums)   # index of the maximum element
nums[which.min(nums)]  # the actual minimum element
nums[which.max(nums)]  # the actual maximum element

c(first=1, second=2, third=3, fourth=4, fifth=5)
positions <- c(1, 2, 3, 4, 5)
names(positions) 
names(positions) <- c("first", "second", "third", "fourth", "fifth")
positions
names(positions)
positions[c("second", "fourth")]



