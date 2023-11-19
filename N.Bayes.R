# Libraries
library(naivebayes)
library(dplyr)
library(ggplot2)
library(psych)

# Data
data <- read.csv(file.choose(), header = T)
str(data)
xtabs(~Species+SepalLengthCm, data = data)
data$SepalLengthCm <- as.factor(data$SepalLengthCm)
data$Species <- as.factor(data$Species)

# Visualization
pairs.panels(data[-1])
data %>%
  ggplot(aes(x=Species, y=SepalLengthCm, fill = Species)) +
  geom_boxplot() +
  ggtitle("Box Plot")

data %>% ggplot(aes(x=SepalLengthCm, fill = Species)) +
  geom_density(alpha=0.8, color= 'black') +
  ggtitle("Density Plot")

# Data Partition
set.seed(1234)
ind <- sample(2, nrow(data), replace = T, prob = c(0.8, 0.2))
train <- data[ind == 1,]
test <- data[ind == 2,]

# Naive Bayes Model
model <- naive_bayes(Species ~ ., data = train, usekernel = T)
model

train %>%
  filter(Species == "1") %>% 
  summarise(mean(PetalWidthCm), sd(PetalWidthCm))

plot(model)

# Predict
p <- predict(model, train, type = 'prob')
head(cbind(p, train))

# Confusion Matrix - train data
p1 <- predict(model, train)
(tab1 <- table(p1, train$Species))
1 - sum(diag(tab1)) / sum(tab1)

# Confusion Matrix - test data
p2 <- predict(model, test)
(tab2 <- table(p2, test$Species))
1 - sum(diag(tab2)) / sum(tab2)

