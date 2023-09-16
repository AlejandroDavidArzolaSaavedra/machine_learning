from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def compare_linear_regression(X, y):
    # Create a linear regression model using scikit-learn
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Coefficients from the scikit-learn model
    coef_skl = round(model.coef_[0], 6)
    intercept_skl = round(model.intercept_, 6)

    y_lineal_regression = lineal_regression_manually(X,y)
    ax_lineal_regression = round(y_lineal_regression['slope (a)'],6)
    b_lineal_regression = round(y_lineal_regression['intercept (b)'],6)
    
    # Visualize the results
    plt.scatter(X, y, label='Actual Data', color='blue')
    plt.plot(X, y_pred, label='Predictions (scikit-learn)', color='red')
    plt.xlabel('Feature')
    plt.ylabel('Target Variable')
    plt.legend()
    plt.xlim(0, 21)  # Ajusta el rango del eje x
    plt.ylim(0, 24)  # Ajusta el rango del eje y (ajusta según tus datos)
    plt.show()

    # Check if the results are equal
    coef_equal = coef_skl == ax_lineal_regression
    intercept_equal = intercept_skl == b_lineal_regression

    return {
        'Coefficient (scikit-learn)': coef_skl,
        'Intercept (scikit-learn)': intercept_skl,
        'Coefficient (manual)': ax_lineal_regression,
        'Intercept (manual)': b_lineal_regression,
        'Coefficients Match': coef_equal,
        'Intercepts Match': intercept_equal
    }


"""
    Method that  calculate the lineal regresion of two arrays
"""
def lineal_regression_manually(X, y):
    sum_xy = 0
    sum_x = 0
    sum_y = 0
    sum_x_squared = 0
    n = len(X)

    # Calculate sums of relevant values
    for i in range(n):
        sum_xy = sum_xy +  X[i] * y[i]
        sum_x =  sum_x + X[i]
        sum_y =  sum_y + y[i]
        sum_x_squared = sum_x_squared + X[i] ** 2

    # Calculate the coefficients (a and b) of the linear equation -> Y = ax + b
    
    a = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    b = (sum_x_squared * sum_y - sum_xy * sum_x) / (n * sum_x_squared - sum_x ** 2)

    return {'slope (a)': a[0], 'intercept (b)': b[0]}