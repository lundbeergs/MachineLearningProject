from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt

x_test_regression = np.load('X_test_regression1.npy')
x_train_regression = np.load('X_train_regression1.npy')
y_train_regression = np.load('y_train_regression1.npy')

model_linear_regression = LinearRegression()

model_linear_regression.fit(x_train_regression, y_train_regression)

y_pred = model_linear_regression.predict(x_test_regression)

np.save('y_pred', y_pred)
