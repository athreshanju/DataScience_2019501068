setwd("C:\\Users\\Chowdary\\Desktop\\Final exam\\Final exam")
liver = read.csv("Liver_data.csv", header = FALSE, col.names = c("mcv", "alkphos", "sgpt", "sgot", "gammagt", "drinks","selector"))
liver$selector <- as.factor(liver$selector)
liver$drinks <- cut(liver$drinks, breaks = c(0, 5,10,15,20), 
                    labels = c('C1', 'C2', "C3", 'C4'), right = FALSE)
liver <- na.omit(liver)
train = subset(liver, liver$selector == 1)
test = subset(liver, liver$selector == 2)
x_train <- subset(train, select = -c(selector, drinks))
x_test <- subset(test, select = -c(selector, drinks))
library(class)
y_train = train[,6, drop = TRUE]
y_test = test[,6, drop = TRUE]
library(e1071)
fit = svm(x_train, y_train)
1-sum(y_train==predict(fit,x_train))/length(y_train)   
fit = svm(x_test, y_test)
1-sum(y_test==predict(fit,x_test))/length(y_test)    
