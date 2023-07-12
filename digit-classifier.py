import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist  
(x_train, y_train),(x_test, y_test) = mnist.load_data()  

x_train = tf.keras.utils.normalize(x_train, axis=1)  
x_test = tf.keras.utils.normalize(x_test, axis=1) 

model = tf.keras.models.Sequential()  
model.add(tf.keras.layers.Flatten())  
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  

model.fit(x_train, y_train, epochs=10)  

val_loss, val_acc = model.evaluate(x_test, y_test)  
print("\nloss =", val_loss)  
print("precision =", val_acc)  

y_predicted = model.predict(x_test)
np.argmax(y_predicted[0])

y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:1]

import os
from PIL import Image

# Preparation of the folder containing the images
image_folder = 'C:\Users\theog\404-ctf\poeme\images'
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# Initialize the list to store the images
images = []

for image_file in image_files:
    # Open each image, convert it to grayscale, and resize it to 28x28 pixels
    image = Image.open(os.path.join(image_folder, image_file)).convert('L').resize((28, 28))
    
    # Convert the image to a numpy array and normalize it (similar to what we did for the training data)
    image = tf.keras.utils.normalize(np.array(image), axis=1)
    
    # Add the image to our list
    images.append(image)

# Convert the list of images into a numpy array
images = np.array(images)

# Predict the classes for each image
predictions = model.predict(images)

# Retrieve the index of the predicted class for each image
predicted_labels = [np.argmax(p) for p in predictions]

print(predicted_labels)
