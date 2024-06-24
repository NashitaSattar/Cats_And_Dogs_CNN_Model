"""This is a binary image classification model using RGB images which will predict if the image is of a cat or a dog.
This uses Keras and TensorFlow"""

import numpy as np
import matplotlib.pyplot as plt
import random
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras import Sequential  # Import Sequential model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten  # Import layers of CNN
from keras import Input  # Import Input layer

# Load Dataset. We have 1000 images of cats and 1000 images of dogs, so 2000 in total
x_train = np.loadtxt('input.csv', delimiter=',')
y_train = np.loadtxt('labels.csv', delimiter=',')

x_test = np.loadtxt('input_test.csv', delimiter=',')
y_test = np.loadtxt('labels_test.csv', delimiter=',')

# Reshape images into 100x100 pixels with 3 channels (RGB)
x_train = x_train.reshape(len(x_train), 100, 100, 3)
y_train = y_train.reshape(len(y_train), 1)  # Reshape labels into 1-dimensional array

x_test = x_test.reshape(len(x_test), 100, 100, 3)
y_test = y_test.reshape(len(y_test), 1)  # Reshape labels into 1-dimensional array

# The values in the array were initially between 0 and 255. We want to scale them from 0 to 1 to train our model
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the model architecture. The Sequential model means that the layers are stacked in a sequence
model = Sequential([
    Input(shape=(100, 100, 3)),  # Input layer that defines the expected shape of the image
    Conv2D(32, (3, 3), activation='relu'),  # First convolutional layer with 32 filter of size 3*3.
    MaxPooling2D((2, 2)),  # First max pooling layer with filter size of 2x2

    Conv2D(32, (3, 3), activation='relu'),  # Second convolutional layer with 32 filters of size 3*3
    MaxPooling2D((2, 2)),  # Second max pooling layer with filter size of 2x2

    Flatten(),  # Flatten the 3D outputs to 1D
    Dense(64, activation='relu'),  # Fully connected layer with 64 neurons
    Dense(1, activation='sigmoid')  # Output layer with 1 neuron and sigmoid activation for binary classification
])

# Compile the model with binary cross-entropy loss, Adam optimizer, and accuracy metric for back propagation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with training data for 5 epochs and batch size of 64
model.fit(x_train, y_train, epochs=5, batch_size=64)

# Evaluate the loss and accuracy of model with test data
model.evaluate(x_test, y_test)

# Randomly select an index from the test set
index = random.randint(0, len(y_test))

# Display the image corresponding to the selected index
plt.imshow(x_test[index])
plt.show()

# Predict the class of the selected image
y_prediction = model.predict(x_test[index].reshape(1, 100, 100, 3))
y_prediction = y_prediction > 0.5  # Convert prediction to binary (True if greater than 0.5, else False)

# Determine the predicted class
if y_prediction == 0:
    prediction = "dog"
else:
    prediction = "cat"

# Print the predicted class
print("Our model says it is a:", prediction)
