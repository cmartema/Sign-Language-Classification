
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split

import keras
from keras.utils  import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

class SignLanguage:
    def __init__(self):
        self.model = None

        self.data = {
            "train": None,
            "test" : None
        }
        self.create_model()

    def create_model(self):
        """
        Create a CNN model and save it to self.model
        """
        # Define the model
        model = Sequential()

        # Add the first convolutional layer
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

        # Add a max pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Add a second convolutional layer
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))

        # Add another max pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # Flatten the output before the fully connected layers
        model.add(Flatten())

        # Add a fully connected layer neurons
        model.add(Dense(128, activation='relu'))

        # Add a dropout layer to reduce overfitting
        model.add(Dropout(0.5))

        # Output layer with as many neurons as classes
        model.add(Dense(num_classes, activation='softmax'))

        # Compile the model with categorical_crossentropy
        model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model

    def prepare_data(self, images, labels):
        """
        Use this method to normalize the dataset and split it into train/test.
        Save your data in self.data["train"] and self.data["test"] as a tuple
        of (images, labels)

        :param images numpy array of size (num_examples, 28*28)
        :param labels numpy array of size (num_examples, )
        """
        # TODO : split into training and validation set
        # TODO : reshape each example into a 2D image (28, 28, 1)

        images = images.reshape(-1, 28, 28, 1)
        images = images/255.0
        oneHotLabel= to_categorical(labels)
        xTrain, xTest, yTrain, yTest = train_test_split(images, oneHotLabel, test_size=0.2, train_size=0.8)

        self.data = {
            "train": (xTrain, yTrain),
            "test" : (xTest, yTest),
        }

    def train(self, batch_size:int=128, epochs:int=50, verbose:int=1):
        """
        Use model.fit() to train your model. Make sure to return the history for a neat visualization.

        :param batch_size The batch size to use for training
        :param epochs     Number of epochs to use for training
        :param verbose    Whether or not to print training output
        """

        x_train, y_train = self.data["train"]
        x_test, y_test = self.data["test"]

        history = self.model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose,
                                 validation_data=(x_test, y_test))

        return history

    def predict(self, data):
        """
        Use the trained model to predict labels for test data.

        :param data: numpy array of test images
        :return a numpy array of test labels. array size = (num_examples, )
        """

        data = data.reshape(-1, 28, 28, 1).astype('float32') / 255
        predictions = self.model.predict(data)
        labels = np.argmax(predictions, axis=1)

        return labels

        # Don't forget to normalize the data in the same way as training data
        # self.model.predict() and np.argmax( , axis=1) might help

    def visualize_data(self, data):
        """
        Visualizing the hand gestures

        :param data: numpy array of images
        """
        if data is None: return

        nrows, ncols = 5, 5
        fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10), sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0, hspace=0)

        for i in range(nrows):
            for j in range(ncols):
                axs[i][j].imshow(data[0][i*ncols+j].reshape(28, 28), cmap='gray')
        plt.show()

    def visualize_accuracy(self, history):
        """
        Plots out the accuracy measures given a keras history object

        :param history: return value from model.fit()
        """
        if history is None: return

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title("Accuracy")
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train','test'])
        plt.show()
