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

