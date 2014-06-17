library(caret)
library(ggplot2)

set.seed(12345)
# Load the training dataset. Replace blanks with NA
tr1 <- read.csv("pml-training.csv",header=TRUE, na.strings=c("","NA"))

# Remove columns with at least one NA in it
df <- tr1[,colSums(is.na(tr1))==0]
# Remove time stamps etc. In particular, remove the first column. There is a strong correlation
# between the row number and the classe variable !
df  <- df[,c(-1, -2,-3,-4,-5,-6,-7)]

# Randomize the order
df  <- df[sample(nrow(df)),]
# Use this to check random order and also to make sure that index is gone: head(df[,c(1:5,53)])

inTrain <- createDataPartition(y=df$classe,p=0.75,list=FALSE)
training <- df[inTrain,]
testing <- df[-inTrain,]

# Applicaton to the training set
preProc <- preProcess(training[,-53], method="pca", pcaComp=8)
trainPC <- predict(preProc,training[,-53])
modelFit <- train(training$classe~., method="knn", data=trainPC)

qplot(x=trainPC[,1],y=trainPC[,2],color=classe,data=training)
qplot(x=1:nrow(training),y=trainPC[,1],color=classe,data=training)

# Apply preProc from the training set to the testing set
testPC <- predict(preProc,testing[,-53])
qplot(x=testPC[,1],y=testPC[,2],color=classe,data=testing)
confusionMatrix(testing$classe,predict(modelFit,testPC))

#---------------------------------------------------------------------------------#

# Load the testing data set.
tr2 <- read.csv("pml-testing.csv",header=TRUE, na.strings=c("","NA"))

df2 <- tr2[,colSums(is.na(tr2))==0]
df2 <- df2[,c(-1, -2,-3,-4,-5,-6,-7)]

actualPC <- predict(preProc,df2[,-53])
A <-  predict(modelFit,actualPC)
