import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Flatten
from keras import Model
from keras.callbacks import Callback
from sklearn.metrics import classification_report
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.model_selection import train_test_split


class TrainingCallback(Callback):
    def __init__(self, X_test, y_test):
        super(TrainingCallback, self).__init__()
        self.X_test = X_test
        self.y_test = y_test

    def on_epoch_begin(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_test)
        y_pred_bool = np.argmax(y_pred, axis=1)
        print(classification_report(self.y_test, y_pred_bool))

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_test)
        y_pred_bool = np.argmax(y_pred, axis=1)
        confusion_matrix(self.y_test, y_pred_bool)


def main():
    iris = datasets.load_iris()

    x, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

    input = Input(shape=4)
    flatten = Flatten()(input)
    dense_1 = Dense(100, activation='relu')(flatten)
    dense_2 = Dense(100, activation='relu')(dense_1)
    output = Dense(10, activation='softmax')(dense_2)

    model = Model(inputs=[input], outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20,
                        callbacks=[TrainingCallback(X_test, y_test)])

    plt.title('Loss without normalisation on Iris dataset')
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend()
    plt.show()

    # with normalisation
    X_train = (X_train - X_train.mean()) / X_train.std()
    X_test = (X_test - X_test.mean()) / X_test.std()
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20,
                        callbacks=[TrainingCallback(X_test, y_test)])

    plt.title('Loss with normalisation on Iris dataset')
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend()
    plt.show()
