<h1 align="center">Linear Regression Comparison</h1>

The `compare_linear_regression` script demonstrates the comparison between linear regression models created using scikit-learn and manual implementation using NumPy. This provides insights into the consistency of results and allows you to verify the accuracy of your linear regression calculations.

## How it Works 🤖

The script performs the following steps:

1. **Scikit-Learn Linear Regression:**
   - Utilizes the `LinearRegression` class from scikit-learn to create a linear regression model.
   - Fits the model to the input data (`X` and `y`).
   - Obtains coefficients and intercept from the scikit-learn model.

2. **Manual Linear Regression:**
   - Implements a manual linear regression calculation using NumPy.
   - Visualizes the linear regression predictions from both scikit-learn and NumPy.

3. **Visualization:**
   - Plots the actual data, predictions from scikit-learn, and predictions from NumPy.
   - Adjusts the plot limits based on your data.

4. **Comparison:**
   - Compares the coefficients and intercepts from both approaches.
   - Checks if the results match.

