## Data Structures in R - Data Frames

# TIP - You can select and execute the lines you want to or you can execute
# the whole file and see the different outputs. Recommended way is to 
# execute it section-wise.


## Creating Data Frames

df <- data.frame(
  real.name = c("Bruce Wayne", "Clark Kent", "Slade Wilson", "Tony Stark", "Steve Rogers"),
  superhero.name = c("Batman", "Superman", "Deathstroke", "Iron Man", "Capt. America"),
  franchise = c("DC", "DC", "DC", "Marvel", "Marvel"),
  team = c("JLA", "JLA", "Suicide Squad", "Avengers", "Avengers"),
  origin.year = c(1939, 1938, 1980, 1963, 1941)
)

df
class(df)
str(df)
rownames(df)
colnames(df)
dim(df)

head(mtcars)


## Operating on Data Frames

df[2:4,]
df[2:4, 1:2]
subset(df, team=="JLA", c(real.name, superhero.name, franchise))
subset(df, team %in% c("Avengers","Suicide Squad"), c(real.name, superhero.name, franchise))


df1 <- data.frame(
  id = c('emp001', 'emp003', 'emp007'),
  name = c('Harvey Dent', 'Dick Grayson', 'James Bond'),
  alias = c('TwoFace', 'Nightwing', 'Agent 007')
)

df2 <- data.frame(
  id = c('emp001', 'emp003', 'emp007'),
  location = c('Gotham City', 'Gotham City', 'London'),
  speciality = c('Split Persona', 'Expert Acrobat', 'Gadget Master')
)

df1
df2

rbind(df1, df2)   # not possible since column names don???t match
cbind(df1, df2)
merge(df1, df2, by="id")

