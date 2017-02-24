## Controlling code flow

# TIP - You can select and execute the lines you want to or you can execute
# the whole file and see the different outputs. Recommended way is to 
# execute it section-wise.


## Working with if, if-else and ifelse

num = 5
if (num == 5){
  cat('The number was 5')
}

num = 7
if (num == 5){
  cat('The number was 5')
} else{
  cat('The number was not 5')
}

if (num == 5){
  cat('The number was 5')
} else if (num == 7){
  cat('The number was 7')
} else{
  cat('No match found')
}

ifelse(num == 5, "Number was 5", "Number was not 5")


## Working with switch

switch(
  "first",
  first = "1st",
  second = "2nd",
  third = "3rd",
  "No position"
)

switch(
  "third",
  first = "1st",
  second = "2nd",
  third = "3rd",
  "No position"
)

switch(
  "fifth",
  first = "1st",
  second = "2nd",
  third = "3rd",
  "No position"
)


## Loops

# for loop
for (i in 1:10){
  cat(paste(i," "))
}

sum = 0
for (i in 1:10){
  sum <- sum + i
}
sum

# while loop
count <- 1
while (count <= 10){
  cat(paste(count, " "))
  count <- count + 1
}

# repeat infinite loop 
count = 1
repeat{
  cat(paste(count, " "))
  if (count >= 10){
    break  # break off from the infinite loop
  }
  count <- count + 1
}

