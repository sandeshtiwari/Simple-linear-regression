# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
# Matrix of independent variable
X = dataset.iloc[:, :-1].values
# Vector of dependent variable
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Fitting Simple Linear Regression to the Training set
# The regressor is the trained machine
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# Training the regressor
regressor.fit(X_train, y_train)

# Predicting the Test set results- making vector with predicted salaries
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color="red")
# Drawing a line with the traing independent matrix and the predicted dependent vector of the training data
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs Experience(Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color="red")
# Drawing a line with the traing independent matrix and the predicted 
# dependent vector of the training data. The regressor is trained so the line will be the same
# So the X_test and X_train will give the same line for the regressor
plt.plot(X_test, regressor.predict(X_test), color="blue")
plt.title("Salary vs Experience(Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
