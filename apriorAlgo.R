# Reading the data file
mydata<-read.csv("C:\\Users\\hp\\Documents\\transaction data.csv",header=T)
head(mydata)
mydata$Transaction.Id <- as.factor(mydata$Transaction.Id)
summary(mydata)
data <- list(
  c("bread","cheese","egg","juice"),
  c("bread","cheese","juice"),
  c("bread","milk","yogurt"),
  c("bread","juice","milk"),
  c("cheese","juice","milk")
)
#apriori algorithm
library(arules)
rules <- apriori(mydata)
summary(rules)
data <- as(data, "transactions")
inspect(data)
basket <- as(data, "tidLists")
inspect(basket)

# Rules with specified parameter valus
rules <- apriori(basket,parameter = list(minlen=2, maxlen=10,supp=.5, conf=.75))
inspect(rules)

# Sorting stuff out
rules<-sort(rules, by="confidence", decreasing=TRUE)
rules <- apriori(basket, parameter = list(supp = 0.5, conf = 0.75,maxlen=3))
rules_conf <- sort (rules, by="confidence", decreasing=TRUE) 
inspect(head(rules_conf))

# Redundancies
redundant <- is.redundant(rules, measure="confidence")
which(redundant)
rules.pruned <- rules[!redundant]
rules.pruned <- sort(rules.pruned, by="lift")
inspect(rules.pruned)

# Targeting Items
rules<-apriori(data=basket, parameter=list(supp=0.001,conf = 0.15,minlen=2), 
               appearance = list(default="rhs",lhs="bread"),
               control = list(verbose=F))
rules<-sort(rules, decreasing=TRUE,by="confidence")
inspect(rules[1:4])

# Visualization
plot(rules)
plot(rules,method="graph",control=list(verbose=F))
plot(rules[1:4], method = "paracoord", control = list(reorder = TRUE))
arulesViz::plotly_arules(rules, method="matrix", measure=c("support","confidence"))
