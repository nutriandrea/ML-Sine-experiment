# ML-Sine-experiment

In this experiment, we will train a Neural Network (NN) to predict a sinewave signal with slight noise. The steps involved are setting up the development environment, preparing data, defining and training the NN, and evaluating the model.

## Table of Contents
- [Setup Development Environment](#setup-development-environment)
- [Data Preparation](#data-preparation)
- [Define and Train NN](#define-and-train-nn)
- [Evaluate and Convert Model](#evaluate-and-convert-model)

## Setup Development Environment

First, we must set up the NN model's development environment. We need to install the Python toolchain on our computers and install the TensorFlow package for your Python interpreter. 

1. Use the Installer program to install Python on your computer.
2. Open the CMD Console and type the following command for better network connection:
    ```sh
    python –m pip install tensorflow –i https://pypi.tuna.tsinghua.edu.cn/simple
    ```

## Data Preparation

1. Generate 1000 samples of a sinewave.
2. Add slight noise to your samples.
3. Split the 1000 samples into three parts:
    - Train Set (80%)
    - Validate Set (10%)
    - Test Set (10%)

Example code to generate and split the data:

```python
import numpy as np

# Generate 1000 samples of a sinewave
x = np.linspace(0, 2 * np.pi, 1000)
y = np.sin(x) + np.random.normal(0, 0.1, x.shape)

# Split the data
train_size = int(0.8 * len(x))
validate_size = int(0.1 * len(x))

x_train, y_train = x[:train_size], y[:train_size]
x_validate, y_validate = x[train_size:train_size + validate_size], y[train_size:train_size + validate_size]
x_test, y_test = x[train_size + validate_size:], y[train_size + validate_size:]
```

## Define and Train NN
1. Add layers to your model.
2. Set some hyperparameters for your model (Batch Size, Epoch).
3. Launch training.
   
Example code to define and train the model:
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Define the model
model = Sequential([
    Dense(50, activation='relu', input_shape=(1,)),
    Dense(50, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Set hyperparameters
batch_size = 32
epochs = 100

# Train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_validate, y_validate))
```

## Evaluate and Convert Model
1. Make a prediction with your model and compare it with the Test Dataset.
2. Quantize your model, convert it, and save it into a .tflite file.
3. Convert your model to a C/C++ file.
4. Example code to evaluate and convert the model:

Example code to evaluate and convert the model:
```python
# Evaluate the model
loss = model.evaluate(x_test, y_test)
print(f'Test Loss: {loss}')

# Make predictions
y_pred = model.predict(x_test)

# Convert to TFLite Model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

# Convert TFLite model to C/C++ source file (example using xxd)
!xxd -i model.tflite > model.cc
```

## Additional Functions
1. Graph the Training History
- The training history graph shows the trend of validation loss and training loss over epochs. It helps visualize how well the model is learning from the training data and if it's overfitting or underfitting.

2. Compare Effectiveness
- Compare the effectiveness between the optimized model and the original one by evaluating metrics such as loss on the test dataset and comparing prediction accuracy.

## Conclusion
This README.md file provides a step-by-step guide to setting up the development environment, preparing the data, defining and training the NN, and evaluating and converting the model. Follow these instructions to successfully complete the ML-Sine-experiment.




