import keras
from keras.applications import vgg16
from keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras import Model
from keras import regularizers
from keras.datasets import cifar10
import ssl
from matplotlib.pyplot import plot as plt


# resolves problem with expired certificate for cifar10
ssl._create_default_https_context = ssl._create_unverified_context


def main():
    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    X_train, X_test = X_train / 255.0, X_test / 255.0

    # importujemy tylko splotową część sieci bez ostatnich warst klasyfikacyjnych
    base_model = vgg16.VGG16(input_shape=[32, 32, 3], weights="imagenet", include_top=False)

    # Definiujemy własne warstwy klasyfikacyjne
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(2096, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = Dense(2096, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    predictions = Dense(1, activation='softmax')(x)

    # tworzymy model podając jako wejście wejście pre-trenowanego modelu i nasze warswy klasyfikacyjne jako wyjście
    model = Model(inputs=base_model.input, outputs=predictions)

    # wyłączamy trening warst pre-trenowanego modelu
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'categorical_accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

    plt.title('Loss')
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.legend()

    plt.plot(history.history['acc'], label='acc')
    plt.plot(history.history['val_acc'], label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
