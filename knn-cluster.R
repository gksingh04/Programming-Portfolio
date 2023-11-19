# Libraries
library(caret)
library(pROC)
library(mlbench)

# Clustering
WNew <- iris


# Knn Clustering Technique

library(class)
library(gmodels)
WNew[is.na(WNew)] <- 0
WSmallSet<-WNew[1:100,]
WTestSet<-WNew[100:150,] # testing set
WLabel<-c(WNew[1:100,5]) # training set
wTestLabel<-c(WNew[100:150,5])
kWset1 <- knnSet <- knn(WSmallSet,WTestSet,WLabel,k=3)
CTab<-CrossTable(x = wTestLabel, y = kWset1,prop.chisq=FALSE)

library(plyr)
library(ggplot2)
set.seed(123)

# Create training and testing data sets
idx = sample(1:nrow(iris), size = 100)
train.idx = 1:nrow(iris) %in% idx
test.idx =  ! 1:nrow(iris) %in% idx

train = iris[train.idx, 1:4]
test = iris[test.idx, 1:4]

# Get labels
labels = iris[train.idx, 5]

# Do knn
fit = knn(train, test, labels)
fit

# Create a dataframe to simplify charting
plot.df = data.frame(test, predicted = fit)

# Use ggplot
# 2-D plots example only
# Sepal.Length vs Sepal.Width

# First use Convex hull to determine boundary points of each cluster
plot.df1 = data.frame(x = plot.df$Sepal.Length, 
                      y = plot.df$Sepal.Width, 
                      predicted = plot.df$predicted)

find_hull = function(df) df[chull(df$x, df$y), ]
boundary = ddply(plot.df1, .variables = "predicted", .fun = find_hull)

ggplot(plot.df, aes(Sepal.Length, Sepal.Width, color = predicted, fill = predicted)) + 
  geom_point(size = 5) + 
  geom_polygon(data = boundary, aes(x,y), alpha = 0.5)

