import tensorflow as td
from tensorflow import keras
import numpy as np
from pathlib import Path


model_file_name = "imdb_model.h5"
vocabulary_size = 88000
# get the data
data = keras.datasets.imdb
# split the data into test and train
(train_data, train_labels), (test_data, test_label) = data.load_data(num_words=vocabulary_size)

# get the word indexes to be able to map the data, that is numbers, to readable words
word_index = data.get_word_index()
# map the tuples resulted from the word index into a list
# keep the first 4 position (the v+3) for special tags
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# pad the data to have the same length
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data, value=word_index["<PAD>"], padding="post", maxlen=250
)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data, value=word_index["<PAD>"], padding="post", maxlen=250
)

# reverse the word index. Why??? Idk
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


def imdb_proc():

    model_file_path = Path(model_file_name)
    if model_file_path.exists():
        model = load_model()
    else:
        model = train_model()
    text_model(model)


def load_model():
    print("loading the model from file")
    return keras.models.load_model(model_file_name)


def train_model():
    print("training new model")
    # define the model
    model = keras.Sequential()
    model.add(keras.layers.Embedding(vocabulary_size, 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))

    model.summary()

    # set the model parameters
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    # get validation data

    x_val = train_data[:10000]
    x_train = train_data[10000:]
    y_val = train_labels[:10000]
    y_train = train_labels[10000:]
    fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)
    results = model.evaluate(test_data, test_label)
    print(results)
    model.save("imdb_model.h5")
    return model


def text_model(model):
    prediction_result = model.predict([test_data])
    accurate = 0
    for i in range(10):
        print(clear_review(decode_review(test_data[i], reverse_word_index)))
        prediction = round(prediction_result[i][0])
        print("predicted: " + str(prediction))
        print("actual: " + str(test_label[i]))
        if prediction == test_label[i]:
            print("This was accurate !!!!!!!!!!!!!!!!")
            accurate = accurate + 1
        else:
            print(" =======> !!!!!!!!!!!!!!!!This was not accurate ")
    print("Resulted accuracy: ", accurate, " out of 10")


def decode_review(text, reverse_word_index):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


def clear_review(review):
    return review.replace("<PAD>", "").replace("<UNK>", "")
