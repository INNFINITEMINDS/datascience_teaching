


# load the training set
training = read.table('~/Documents/teaching/bme_datascience/datascience_teaching/train_regression.csv',header = TRUE,sep=',')

# fit a linear model to predict y from x using the training set
fit1 <- lm(y~x, data = training)

# examine R^2 for the model fit on the training data. This can be done two ways:
# 1. Using the summary() function
print(summary(fit1))
# 2. Using the predict() function with the values of x from the training data as input, then manually calculate R^2
y_predicted_with_train_only = predict(fit1,newdata = training)
r2 = 1 - sum((y_predicted_with_train_only-training$y)^2)/sum((training$y - mean(training$y))^2)
# notice that the R^2 returned from both methods is the same.

# Now let's evaluate the performance of the trained model on the test set.
# load the test set
testing = read.table('~/Documents/teaching/bme_datascience/datascience_teaching/test_regression.csv',header = TRUE,sep=',')

# using the previously fit model, let's predict y values given the x values in the test set
y_predicted_for_test = predict(fit1, newdata = testing)
# Here, we generate predicted y values, rather than a new linear model, so we can't use the summary() function to get R^2.
# So, we calculate it manually using the test set y values:
r2 = 1 - sum((y_predicted_for_test-testing$y)^2)/sum((testing$y - mean(testing$y))^2)

# notice the performance on the test set is actually better than the training set. This is almost never true for 
# real data, but it happens to be the case for this toy data.


# Let's repeat the process for a dataset containing more than one predictor variable.
# load the training and test sets
multi_training = read.table('~/Documents/teaching/bme_datascience/datascience_teaching/train_3dimensions_regression.csv',header = TRUE,sep=',')
multi_testing = read.table('~/Documents/teaching/bme_datascience/datascience_teaching/test_3dimensions_regression.csv',header = TRUE,sep=',')

# fit a linear model to the training data. The variable names are x, x2, and x3
fit2 <- lm(y~x+x2+x3, data = multi_training)
# check the R^2 value
print(summary(fit2))

# Now evaluate the model using the test data, just as we did before
multi_y_predicted_for_test = predict(fit2, newdata = multi_testing)
r2 = 1 - sum((multi_y_predicted_for_test-multi_testing$y)^2)/sum((multi_testing$y - mean(multi_testing$y))^2)

# Given the model trained using 3 predictor variables, we might now be interested in determining which variable contributes most to model accuracy.
# For the lm function in R, variable importance (or feature importance) is assessed using the t-statistic for each variable. This is returned with 
# the summary() function, e.g.:
print(summary(fit2))

# Once the model is fitted using training data, it contains all the information needed to assess variable importance. Keep in mind that variable importance has
# little meaning if the model performs poorly on the test set.
# The caret package creates an interface for many machine learning methods; one function that is particularly useful is varImp(), which returns
# variable importances for whichever model is passed into the function. Had we used varImp() for our linear model, rather than summary(), it would have
# returned the same t-statistic value as summary(). For other machine learning models, variable importance is calculated differently.

