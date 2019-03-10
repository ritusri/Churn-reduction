rm(list=ls(all=T))
setwd("C:/Users/anupr/Desktop/churn prediction")


#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')


#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

install.packages("tidyverse")

install.packages("ggplot2")

## Read the data
train = read.csv("Train_data.csv", header = T, na.strings = c(" ", "", "NA"))


##Data Manupulation; convert string categories into factor numeric
for(i in 1:ncol(train)){
  
  if(class(train[,i]) == 'factor'){
    
    train[,i] = factor(train[,i], labels=(1:length(levels(factor(train[,i])))))
    
  }
}


#BoxPlots - Distribution and Outlier Check
numeric_index = sapply(train,is.numeric) #selecting only numeric

numeric_data = train[,numeric_index]

cnames = colnames(numeric_data)


for (i in 1:length(cnames))
   {
     assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "Churn"), data = subset(train))+ 
              stat_boxplot(geom = "errorbar", width = 0.5) +
              geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                           outlier.size=1, notch=FALSE) +
              theme(legend.position="bottom")+
              labs(y=cnames[i],x="Churn")+
              ggtitle(paste("Box plot of Churner for",cnames[i])))
  }

  
  # ## Plotting plots together
   gridExtra::grid.arrange(gn1,gn4,gn7,gn10,gn13,gn16,ncol=6)
   gridExtra::grid.arrange(gn2,gn5,gn8,gn11,gn14,ncol=5)
   gridExtra::grid.arrange(gn3,gn6,gn9,gn12,gn15,ncol=5)  
   
   # #Remove outliers using boxplot method
    df = train
    train = df

   
     #loop to remove from all variables
     for (i in cnames){
       print(i)
       value = train[,i][train[,i] %in% boxplot.stats(train[,i])$out]
       print(length(value))
       train = train[which(!train[,i] %in% value),]
     }    
    
    
   
    
    #Replace all outliers with NA and impute
    # #create NA 
    #for(i in cnames){
             #value = train[,i][train[,i] %in% boxplot.stats(train[,i])$out]
             #print(length(value))
               #train[,i][train[,i] %in% value] = NA
    #}
    
   
    #train$number.vmail.messages[is.na(train$number.vmail.messages)] = mean(train$number.vmail.messages, na.rm = T)
    #train$total.day.minutes[is.na(train$total.day.minutes)] = mean(train$total.day.minutes, na.rm = T)
    #train$total.day.calls[is.na(train$total.day.calls)] = mean(train$total.day.calls, na.rm = T)
    #train$total.day.charge[is.na(train$total.day.charge)] = mean(train$total.day.charge, na.rm = T)
    #train$total.eve.minutes[is.na(train$total.eve.minutes)] = mean(train$total.eve.minutes, na.rm = T)
    #train$total.eve.calls[is.na(train$total.eve.calls)] = mean(train$total.eve.calls, na.rm = T)
    #train$total.eve.charge[is.na(train$total.eve.charge)] = mean(train$total.eve.charge, na.rm = T)
    #train$total.night.minutes[is.na(train$total.night.minutes)] = mean(train$total.night.minutes, na.rm = T)
    #train$total.night.calls[is.na(train$total.night.calls)] = mean(train$total.night.calls, na.rm = T)
    #train$total.night.charge[is.na(train$total.night.charge)] = mean(train$total.night.charge, na.rm = T)
    #train$total.intl.minutes[is.na(train$total.intl.minutes)] = mean(train$total.intl.minutes, na.rm = T)
    #train$total.intl.calls[is.na(train$total.intl.calls)] = mean(train$total.intl.calls, na.rm = T)
    #train$total.intl.charge[is.na(train$total.intl.charge)] = mean(train$total.intl.charge, na.rm = T)
    #train$number.customer.service.calls[is.na(train$number.customer.service.calls)] = mean(train$number.customer.service.calls, na.rm = T)
    
    
   
    #Correlation Plot 
    corrgram(train[,numeric_index], order = F,
             upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

    
    #Chi-squared Test of Independence
    factor_index = sapply(train,is.factor)
    factor_data = train[,factor_index]    
    
    for (i in 1:4)
    {
      print(names(factor_data)[i])
      print(chisq.test(table(factor_data$Churn,factor_data[,i])))
    }
    #Dimension Reduction
    train = subset(train, 
                          select = -c(total.intl.charge,total.night.charge,total.day.charge))
    
    #Normalisation
    cnames = c("number.vmail.messages","customer.service.calls","total.intl.calls","total.intl.minutes","total.night.calls","total.night.minutes","total.eve.charge",
               "total.eve.calls","total.day.calls","total.day.minutes","number.vmail.messages")
    
    for(i in cnames){
      print(i)
      train[,i] = (train[,i] - min(train[,i]))/
        (max(train[,i] - min(train[,i])))
    }
       
    set.seed(1234)
    train.index = createDataPartition(train$responded, p = .80, list = FALSE)
    train = train[ train.index,]
    test  = train[-train.index,]
    #Develop Model on training data
    C50_model = C5.0(Churn ~., train, trials = 100, rules = TRUE)   
    
    #write rules into disk
    write(capture.output(summary(C50_model)), "c50Rules.txt")
    
    #Lets predict for test cases
    C50_Predictions = predict(C50_model, test[,-17], type = "class")
    
    #Evaluate the performance of classification model
    ConfMatrix_C50 = table(test$responded, C50_Predictions)
    confusionMatrix(ConfMatrix_C50)
    
    
    #False Negative rate
    FNR = FN/FN+TP
    #FN = 310
    #Tp =1824
    #0.1452
    # means our model is good
    
    #Random Forest
    RF_model = randomForest(Churn ~ ., train, importance = TRUE, ntree = 500)
    
    #Logistic Regression
    logit_model = glm(Churn ~ ., data = train, family = "binomial")
    
    #summary of the model
    summary(logit_model)
    
    
    #predict using logistic regression
    logit_Predictions = predict(logit_model, newdata = test, type = "response")
    
    #convert prob
    logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)
    
    
    ##Evaluate the performance of classification model
    ConfMatrix_RF = table(test$Churn, logit_Predictions)
    
    #False Negative rate
   # FNR = FN/FN+TP
   #FN = 640
   #TP = 1554
    
    
   
    