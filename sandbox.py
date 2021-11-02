from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
import numpy as np


def main():
    wine = load_wine()
    X = wine['data']
    y = wine['target']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(13,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30)

    y_prob = model.predict(x_test)
    y_pred = np.argmax(y_prob, axis=1)
    acc_score = accuracy_score(y_test, y_pred)
    if acc_score < 1:
        # ROC
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob[:, 0], pos_label=0)
        plt.plot(false_positive_rate, true_positive_rate)
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    # Training accuracy
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
