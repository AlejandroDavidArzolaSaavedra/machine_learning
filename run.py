from linear_regression.linear_regression import compare_linear_regression
import numpy as np

# First Activity -> Lineal Regression
X = np.arange(1, 21).reshape(-1, 1)
y = np.array([2, 4, 1, 4, 5, 8, 10, 8, 9, 4, 11, 18, 13, 14, 20, 12, 17, 21, 19, 20])
results = compare_linear_regression(X, y)


print(results)