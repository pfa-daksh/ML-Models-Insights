attach(Russia_Real_Estate_Data)
library(ISLR)
library(leaps)#for computing step wise regression
library(naniar)#for missing values
library(IDPmisc) #NaRV omit
library(corrplot)#for correlation
library(dplyr)
library(tidyverse)#data manipulation and visualization
library(caret)#cross-validation methods
data <- Russia_Real_Estate_Data #level is for appartment floor and levels is for numbers of floors in the building
vis_miss(data)
d1 <- na.omit(data) # Method 1 - Remove NA
d1
data2 <- subset(d1, select = -c(street_id,id_region,house_id,date,geo_lat,geo_lon,building_type))
vis_miss(data2)
# Forward Selection Method
regfit_fwd = regsubsets(price~., data = data2, nvmax = 10, method = "forward")
reg_summary_fwd=summary(regfit_fwd)
summary(regfit_fwd)
reg_summary_fwd$adjr2# adjR square values
adj_r2_max = which.max(reg_summary_fwd$adjr2) # maximum R-square
adj_r2_max

coef(regfit_fwd,6)#coefficient
plot(1:30)
plot(reg_summary_fwd$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
points(adj_r2_max, reg_summary_fwd$adjr2[adj_r2_max], col ="red", cex = 2, pch = 20)# to show max adj_r2_max on plot                             
#Variables selected are coming out to be 6 but 3 variables are sufficient for a good Adj Rsq value

# Backward Selection.
regfit_back = regsubsets(price~., data = data2, nvmax = 10, method = "backward")
reg_summary_back=summary(regfit_back)
summary(regfit_back)
reg_summary_back$adjr2# adjR square values
adj_r2_max = which.max(reg_summary_back$adjr2) # maximum R-square
adj_r2_max
coef(regfit_back,6)#coefficient
plot(reg_summary_back$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
points(adj_r2_max, reg_summary_back$adjr2[adj_r2_max], col ="red", cex = 2, pch = 20)# to show max adj_r2_max on plot
#Variables selected are coming out to be 6 but 3 variables are sufficient for a good Adj Rsq value


#Best Subsets(Exhaustive Search)
regfit_best = regsubsets(price~., data = data2, nvmax = 10, method = "exhaustive")
reg_summary_best=summary(regfit_best)
summary(regfit_best)
reg_summary_best$adjr2# adjR square values
adj_r2_max = which.max(reg_summary_best$adjr2) # maximum R-square
adj_r2_max
coef(regfit_best,6)#coefficients and the dimensions to be included.
plot(reg_summary_best$adjr2, xlab = "Number of Variables", ylab = "Adjusted RSq", type = "l")
points(adj_r2_max, reg_summary_best$adjr2[adj_r2_max], col ="red", cex = 2, pch = 20)# to show max adj_r2_max on plot
#Variables selected are coming out to be 6 but 3 variables are sufficient for a good Adj Rsq value

# Selecting Models using cross validation"
data_best_select <- subset(data2, select = c(area,postal_code,rooms,levels,object_type,level,price))
data_4 <- subset(data2, select = c(area,postal_code,rooms,price))

# Inspect the data
sample_n(data_best_select, 4)
# Split the data into training and test set
set.seed(16)#used to generate a sequence of random numbers - it ensures that you get the same result if you start with that same seed each time you run the same process.
training.samples <- data_best_select$price %>% #%>% is called the forward pipe operator in R. It provides a mechanism for chaining commands with a new forward-pipe operator, %>%. This operator will forward a value, or the result of an expression, into the next function call/expression.
createDataPartition(p = 0.7, list = FALSE)#70% is used for training
train.data  <- data_best_select[training.samples, ]
test.data <- data_best_select[-training.samples, ]

# Build the model
model1 <- lm(price ~., data = train.data)
# Make predictions and compute the R2, RMSE and MAE
predictions <- model1 %>% predict(test.data)
data.frame( R2 = R2(predictions, test.data$price),
            RMSE = RMSE(predictions, test.data$price),
            MAE = MAE(predictions, test.data$price))

RMSE(predictions, test.data$price)/mean(test.data$price)# This Gives the prediction error rate.

"Validation set approach for data model built with 3 features from best selection."

# Inspect the data
sample_n(data_4, 4)
# Split the data into training and test set
set.seed(16)#used to generate a sequence of random numbers - it ensures that you get the same result if you start with that same seed each time you run the same process.
training.samples <- data_4$price %>% #%>% is called the forward pipe operator in R. It provides a mechanism for chaining commands with a new forward-pipe operator, %>%. This operator will forward a value, or the result of an expression, into the next function call/expression.
  createDataPartition(p = 0.7, list = FALSE)#70% is used for training
train.data  <- data_4[training.samples, ]
test.data <- data_4[-training.samples, ]
# Build the model
model2 <- lm(price ~., data = train.data)
# Make predictions and compute the R2, RMSE and MAE
predictions <- model2 %>% predict(test.data)
data.frame( R2 = R2(predictions, test.data$price),
            RMSE = RMSE(predictions, test.data$price),
            MAE = MAE(predictions, test.data$price))

RMSE(predictions, test.data$price)/mean(test.data$price)# This Gives the prediction error rate.

"K Fold Cross Validation approach for data model built with features from best selection."
set.seed(16) 
train.control <- trainControl(method = "cv", number = 10)
# Train the model
model4 <- train(price~., data = data_best_select, method = "lm",
                trControl = train.control)
# Summarize the results
print(model4)

"K Fold Cross Validation approach for data model built with 5 best features from best selection."
set.seed(16) 
train.control <- trainControl(method = "cv", number = 10)
# Train the model
model5 <- train(price~., data = data_4, method = "lm",
                trControl = train.control)
# Summarize the results
print(model5)
