<h1 align="center">Perceptron 🧠</h1>

<p align="center">
   <img width="400px" src="https://github.com/AlejandroDavidArzolaSaavedra/machine_learning/assets/90756437/601d4b2a-757e-47be-ad75-8664cc7fb038">
</p>

The `Perceptron` is a Python implementation of a simple perceptron, a basic building block in neural networks. A perceptron is a binary classifier that learns to classify input data into two classes, typically denoted as 1 or -1. It achieves this classification by adjusting its weights based on the input features.

## How it Works 🤖

The perceptron follows a simple learning algorithm:

1. **Initialization:**
   - Learning Rate (`eta`): Determines the step size in adjusting weights during training.
   - Number of Iterations (`n_iter`): Sets the number of passes over the training dataset.
   - Random State (`random_state`): Seed for the random number generator, influencing initial weight values.

2. **Training:**
   - The perceptron is trained using the provided input features (`x`) and target labels (`y`).
   - During training, the weights are adjusted iteratively to minimize classification errors.

3. **Prediction:**
   - Once trained, the perceptron can predict the class label for new input data.

## Usage 🚀

To use the `Perceptron` class, follow these steps:

1. Import the necessary libraries:

   ```python
   import numpy as np
