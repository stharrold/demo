###############################################################
#Chapter      :   2 
#Algorithm    :   KMeans
# Description :   The following code clusters IRIS dataset using
#                 kmeans algorithm
###############################################################

iris

# prepare a copy of iris data set
kmean_iris <- iris

#Erase/ Nullify species labels
kmean_iris$Species <- NULL

#apply k-means with k=3
(clusters <- kmeans(kmean_iris, 3)) 

# comparing cluster labels with actual iris  species labels.
table(iris$Species, clusters$cluster)

# plot the clustered points along sepal length and width
plot(kmean_iris[c("Sepal.Length", "Sepal.Width")], col=clusters$cluster,pch = c(15, 16, 17)[as.numeric(clusters$cluster)])

points(clusters$centers[,c("Sepal.Length", "Sepal.Width")], col=1:3, pch=8, cex=4)