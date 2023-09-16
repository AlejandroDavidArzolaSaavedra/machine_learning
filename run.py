from linear_regression.linear_regression import compare_linear_regression
from perceptron.perceptron import Perceptron
import numpy as np
#####################################
# @author: ADAS
# @update: 16/09/2023
#####################################

#####################################
# First Activity -> Lineal Regression
#####################################

# X = np.arange(1, 21).reshape(-1, 1)
# y = np.array([2, 4, 1, 4, 5, 8, 10, 8, 9, 4, 11, 18, 13, 14, 20, 12, 17, 21, 19, 20])
# results = compare_linear_regression(X, y)
# print(results)

#################################
# Second Activity -> Perceptron
#################################


# Paso 1: Crear un conjunto de datos de prueba
X = np.array([[2, 3], [4, 5], [1, 1], [3, 2]])
y = np.array([1, 1, -1, -1])

# Paso 2: Crear una instancia de Perceptron
perceptron = Perceptron(eta=0.1, n_iter=50, random_state=1)

# Paso 3: Ajustar el Perceptron a los datos de entrenamiento
perceptron.fit(X, y)

print("Pesos finales del modelo:")
print(perceptron.w_)
print("Numero de errores en cada iteracion:")
print(perceptron.errors_)

"""
    OUTPUT
    Pesos finales del modelo:    [-0.78375655 -0.20611756  1.59471828]
    Numero de errores en cada iteracion: [1, 2, 3, 2, 3, 2, 2, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
"""
