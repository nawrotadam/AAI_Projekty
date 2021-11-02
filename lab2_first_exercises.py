from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from keras.layers import Dense
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve

LAYERS_SIZES_A = [1, 5, 10, 50, 100, 1000]
LAYERS_QUANTITIES_B = [0, 1, 2, 10]


def main():
    dataset = load_wine()
    X = dataset['data']
    y = dataset['target']

    for k in range(len(LAYERS_SIZES_A)):  # a)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        model = create_model(hidden_layers_size=LAYERS_SIZES_A[k], hidden_layers_quantity=2)
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30)
        plot_data(history)

        y_prob = model.predict(x_test)
        y_pred = np.argmax(y_prob, axis=1)
        acc_score = accuracy_score(y_test, y_pred)
        if acc_score < 1:
            # ROC
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob[:, 0], pos_label=0)
            plt.title("ROC")
            plt.plot(false_positive_rate, true_positive_rate)
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()

    for n in range(len(LAYERS_QUANTITIES_B)):  # b)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
        model = create_model(hidden_layers_size=50, hidden_layers_quantity=LAYERS_QUANTITIES_B[n])
        history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30)
        plot_data(history)

        y_prob = model.predict(x_test)
        y_pred = np.argmax(y_prob, axis=1)
        acc_score = accuracy_score(y_test, y_pred)
        if acc_score < 1:
            # ROC
            false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob[:, 0], pos_label=0)
            plt.title("ROC")
            plt.plot(false_positive_rate, true_positive_rate)
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()


def create_model(hidden_layers_quantity, hidden_layers_size):
    model = Sequential()
    for i in range(hidden_layers_quantity):
        model.add(Dense(hidden_layers_size, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model


def plot_data(history):
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()
