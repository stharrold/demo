## Data Structures in R - Arrays & Matrices

# TIP - You can select and execute the lines you want to or you can execute
# the whole file and see the different outputs. Recommended way is to 
# execute it section-wise.


## Creating Arrays and Matrices

three.dim.array <- array(
  1:32,    # input data
  dim = c(4, 3, 3),   # dimensions
  dimnames = list(    # names of dimensions
    c("row1", "row2", "row3", "row4"),
    c("col1", "col2", "col3"),
    c("first.set", "second.set", "third.set")
  )
)
three.dim.array

mat <- matrix(
  1:24,   # data
  nrow = 6,  # num of rows
  ncol = 4,  # num of columns
  byrow = TRUE  # fill the elements row-wise
)
mat


## Names and Dimensions

dimnames(three.dim.array)
rownames(three.dim.array)
colnames(three.dim.array)
dimnames(mat)
rownames(mat)
rownames(mat) <- c("r1", "r2", "r3", "r4", "r5", "r6")
colnames(mat) <- c("c1", "c2", "c3", "c4")
dimnames(mat)
mat


dim(three.dim.array)
nrow(three.dim.array)
ncol(three.dim.array)
length(three.dim.array)  # product of dimensions
dim(mat)
nrow(mat)
ncol(mat)
length(mat)


## Matrix Operations

mat1 <- matrix(
  1:15,
  nrow = 5,
  ncol = 3,
  byrow = TRUE,
  dimnames = list(
    c("M1.r1", "M1.r2", "M1.r3", "M1.r4", "M1.r5")
    ,c("M1.c1", "M1.c2", "M1.c3")
  )
)
mat1

mat2 <- matrix(
  16:30,
  nrow = 5,
  ncol = 3,
  byrow = TRUE,
  dimnames = list(
    c("M2.r1", "M2.r2", "M2.r3", "M2.r4", "M2.r5"),
    c("M2.c1", "M2.c2", "M2.c3")
  )
)
mat2

rbind(mat1, mat2)
cbind(mat1, mat2)
c(mat1, mat2)

mat1 + mat2   # matrix addition
mat1 * mat2  # element-wise multiplication
tmat2 <- t(mat2)  # transpose
tmat2
mat1 %*% tmat2   # matrix inner product

m <- matrix(c(5, -3, 2, 4, 12, -1, 9, 14, 7), nrow = 3, ncol = 3)
m
inv.m <- solve(m)  # matrix inverse
inv.m
round(m %*% inv.m) # matrix * matrix_inverse = identity matrix






