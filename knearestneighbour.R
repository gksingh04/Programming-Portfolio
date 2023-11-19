# Libraries
library(caret)
library(pROC)
library(mlbench)

#load data
df <- data(iris) 

# see the studcture
head(iris) 

#Generate a random number that is 90% of the total number of rows in dataset.
ran <- sample(1:nrow(iris), 0.9 * nrow(iris)) 

#the normalization function is created
nor <-function(x) { (x -min(x))/(max(x)-min(x))   }

#Run nomalization on first 4 coulumns of dataset because they are the predictors
iris_norm <- as.data.frame(lapply(iris[,c(1,2,3,4)], nor))

summary(iris_norm)

#extract training set
iris_train <- iris_norm[ran,] 

#extract testing set
iris_test <- iris_norm[-ran,] 

#extract 5th column of train dataset because it will be used as 'cl' argument in knn function.
iris_target_category <- iris[ran,5]

#extract 5th column if test dataset to measure the accuracy
iris_test_category <- iris[-ran,5]

#load the package class
library(class)

#run knn function
pr <- knn(iris_train,iris_test,cl=iris_target_category,k=5)
print(pr,digits = 3)

#create confusion matrix
tab <- table(pr,iris_test_category)
print(tab,digits = 3)

#this function divides the correct predictions by total number of predictions that tell us how accurate teh model is.
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)

#Cross Table
library(gmodels)
CrossTable(x = iris_test_category, y = pr, prop.chisq=FALSE)

#plot
plot(iris_test)
plot(iris_train)
plot(pr)
plot(tab)
plot(iris_test_category)
plot(iris_target_category)

