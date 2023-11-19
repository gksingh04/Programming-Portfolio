# Clustering

mydata <- read.csv(file.choose(), header = T)
str(mydata)
head(mydata)
pairs(mydata)

#Scatter Plot
plot(mydata$Height ~ mydata$Weight, data = mydata)

# Normalize 
z = mydata
means = apply(z,2,mean)
print(means,digits = 3)
sds = apply(z,2,sd)
print(sds,digits = 3)
nor = scale(z,center=means,scale=sds)
print(nor,digits = 3)


##calculate distance matrix (default is Euclidean distance)
distance = dist(z)
print(distance,digits = 2)


# Hierarchical agglomerative clustering using default complete linkage 
hc.c <- hclust(distance)
plot(hc.c,hang=-1)

# Hierarchical agglomerative clustering using "average" linkage 
hc.a <- hclust(distance,method="average")
plot(hc.a,hang=-1)

# Cluster membership
member.c <- cutree(hc.c,2)
member.a <- cutree(hc.a,2)
table(member.c,member.a)

#Characterizing clusters 
aggregate(nor,list(member.c),mean)
aggregate(mydata,list(member.c),mean)

#Silhouette PLot
library(cluster)
plot(silhouette(cutree(hc.c,2), distance))

# K-means clustering
kc<-kmeans(z,2)
kc
plot(mydata, col=kc$cluster)

