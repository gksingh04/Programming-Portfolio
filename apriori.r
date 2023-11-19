library(arules)
data <- list(
  c("bread","cheese","egg","juice"),
  c("bread","cheese","juice"),
  c("bread","milk","yogurt"),
  c("bread","juice","milk"),
  c("cheese","juice","milk")
)
data <- as(data, "transactions")
inspect(data)
t1 <- as(data, "tidLists")
inspect(t1)

rules <- apriori(t1, parameter = list(supp = 0.5, conf = 0.75))
inspect(rules)

frequentItems <- eclat(t1, parameter = list(supp = 0.5, maxlen = 15))
inspect(frequentItems)

rules_conf <- sort (rules, by="confidence", decreasing=TRUE) # 'high-confidence' rules.
inspect(head(rules_conf))
