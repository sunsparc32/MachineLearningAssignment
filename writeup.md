Classification with KNN Algorithm
========================================================
I used the humble KNN classifier operating on 8 principal components. I was able to achieve about 92% accuracy on the testing set that I created from the training set. On the actual assignment problems, I was able to get 19 correct, leading to a prediction accuracy of 95%. 

I could have achieved, like many in the formums did, 100 % accuracy using a random forest approach, but I did not do this for three reasons: (1) The knn classifier is simple and easy to understand and I was able to understand its mechanics well. I am not as clear on the internal workings of the random forest classifier and I wanted to stick with something that I understood well. (2) The knn classifier took less than 2 minutes to complete on my iMac and (3) It only took me about five hours total to develop the algorithm and write the necessary R code. 

Data Preprocesing
=====================
First we load the caret package and the ggplot2 package for plotting. Next, we load the training set. Note that blanks are assinged "NA". 

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(ggplot2)

# Load the training dataset. Replace blanks with NA
tr1 <- read.csv("pml-training.csv", header = TRUE, na.strings = c("", "NA"))
```

```
## Warning: cannot open file 'pml-training.csv': No such file or directory
```

```
## Error: cannot open the connection
```

```r

# Remove columns with at least one NA in it
df <- tr1[, colSums(is.na(tr1)) == 0]
```

```
## Error: object 'tr1' not found
```


The data in the training set are arranged sequentially, i.e., all "A" classes are listed first, followed by all "B" classes and so on. This leads to a high correlation between the index variable and classe. This is misleadning and potentially dangerous. For instance, when the {\it roll_belt} variable is plotted against the index number (X), we get the following: 


```r
qplot(X, roll_belt, data = df, color = classe)
```

```
## Error: ggplot2 doesn't know how to deal with data of class function
```


Notice the nice separation with respect to the index variable, X. This is due to the sequential data ordering and surely we cannot expect the actual test to have the classes so ordered. Hence we remove the first column from our analysis and randomize the row order, just to be sure. We also remove some other columns: those with NAs in them, those with the time stamps and so on. There are 160 columns in the raw data and we remove all columns with at least one NA in it. If there are very few predictors, then it may be necessary to do some imputing, but with 160 of them, we can afford to throw some of them away.


```r
# Remove columns with at least one NA in it
df <- tr1[, colSums(is.na(tr1)) == 0]
```

```
## Error: object 'tr1' not found
```

```r
# Remove time stamps etc. In particular, remove the first column. There is a
# strong correlation between the row number and the classe variable !
df <- df[, c(-1, -2, -3, -4, -5, -6, -7)]
```

```
## Error: object of type 'closure' is not subsettable
```

```r

# Randomize the order
df <- df[sample(nrow(df)), ]
```

```
## Error: object of type 'closure' is not subsettable
```

```r
# Use this to check random order and also to make sure that index is gone:
head(df[, c(1:5, 53)])
```

```
## Error: object of type 'closure' is not subsettable
```


