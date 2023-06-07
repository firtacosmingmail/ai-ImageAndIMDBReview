import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def imgProc():
    data = keras.datasets.fashion_mnist
    # split test and train data
    (train_images, train_labels), (test_images, test_labels) = data.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                   'Ankle boot']

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # define the layers of the model ->
    # 3 layers,
    # - 1. the entry layer, the images
    # - 2. the hidden layer,
    # - 3. the result layer, the 10 states
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax")
    ])
    # setup parameters for the model
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # train the model
    model.fit(train_images, train_labels, epochs=5)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print("Tested acc:", test_acc)

    prediction = model.predict([test_images])
    for i in range(5):
        plt.grid(False)
        plt.imshow(test_images[i], cmap=plt.cm.binary)
        plt.xlabel("Actual: " + class_names[test_labels[i]])
        plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
        plt.show()
