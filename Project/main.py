import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import keras
import efficientnet.keras as efn
from keras.callbacks import Callback
from sklearn.metrics import classification_report
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from keras.applications import vgg16
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, GlobalAveragePooling2D
from keras import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from keras import optimizers

DATA_DATASET_PATH = pathlib.Path("../Project/datasets/boy_or_girl/data")


class DataCallback(Callback):
    def __init__(self, validation_generator, model):
        super(DataCallback, self).__init__()
        self.validation_generator = validation_generator
        self.model = model

    def on_epoch_begin(self, epoch, logs=None):
        validation_steps_per_epoch = np.math.ceil(self.validation_generator.samples / self.validation_generator.batch_size)

        predictions = self.model.predict_generator(self.validation_generator, steps=validation_steps_per_epoch)
        predicted_classes = np.argmax(predictions, axis=1)

        true_classes = self.validation_generator.classes
        class_labels = list(self.validation_generator.class_indices.keys())

        report = classification_report(true_classes, predicted_classes, target_names=class_labels)
        print(report)


# split dataset to training and validation data, randomises it also a little
def read_dataset():
    data_gen = ImageDataGenerator(rescale=1. / 255, width_shift_range=0.2, validation_split=0.1, height_shift_range=0.2,
                                  shear_range=0.2, horizontal_flip=True, vertical_flip=True, zoom_range=0.2)

    _training_data = data_gen.flow_from_directory(DATA_DATASET_PATH, target_size=(150, 150), class_mode='binary',
                                                  batch_size=32, subset='training')

    _validation_data = data_gen.flow_from_directory(DATA_DATASET_PATH, target_size=(150, 150), class_mode='binary',
                                                    batch_size=32, subset='validation')

    return _training_data, _validation_data


def inception(_training_data, _validation_data):
    pre_trained_model = InceptionV3(include_top=False, input_shape=(150, 150, 3), weights='imagenet')

    # turn off learning for pre-trained model
    for layers in pre_trained_model.layers:
        layers.trainable = False

    last_layer = pre_trained_model.get_layer('mixed8')
    last_output = last_layer.output

    # our classification layers
    x = tf.keras.layers.Flatten()(last_output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(pre_trained_model.input, x)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='Adam', metrics=['accuracy'])

    return model.fit(_training_data, epochs=2, steps_per_epoch=5, validation_data=_validation_data,
                     callbacks=[DataCallback(_validation_data, model)])


def vggnet(_training_data, _validation_data):
    base_model = vgg16.VGG16(input_shape=(150, 150, 3), weights="imagenet", include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Dense(2096, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = Dense(2096, use_bias=False, kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    predictions = Dense(1, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    return model.fit(_training_data, epochs=2, steps_per_epoch=5, validation_data=_validation_data)


def resnet(_training_data, _validation_data):
    base_model = ResNet50(input_shape=(150, 150, 3), include_top=False, weights="imagenet")
    for layer in base_model.layers:
        layer.trainable = False

    base_model = Sequential()
    base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
    base_model.add(Dense(1, activation='sigmoid'))

    base_model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    return base_model.fit(_training_data, validation_data=_validation_data, steps_per_epoch=5, epochs=2)


def efficientnet(_training_data, _validation_data):
    base_model = efn.EfficientNetB0(input_shape=(150, 150, 3), include_top=False, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)

    predictions = Dense(1, activation="sigmoid")(x)
    model_final = Model(inputs=base_model.input, outputs=predictions)

    model_final.compile(optimizers.RMSprop(lr=0.0001, decay=1e-6), loss='binary_crossentropy', metrics=['accuracy'])
    return model_final.fit(_training_data, validation_data=_validation_data, steps_per_epoch=5, epochs=2)


def plot_results(_history):
    acc = _history.history['accuracy']
    val_acc = _history.history['val_accuracy']
    loss = _history.history['loss']
    val_loss = _history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc=0)
    plt.figure()
    plt.show()

    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend(loc=0)
    plt.figure()
    plt.show()


if __name__ == "__main__":
    # TODO w ostatecznej wersji zastosowac 10 epok po 100 krokow w kazdym z algorytmow

    training_data, validation_data = read_dataset()

    history = inception(training_data, validation_data)
    plot_results(history)

    # history = vggnet(training_data, validation_data)
    # plot_results(history)

    # history = resnet(training_data, validation_data)
    # plot_results(history)

    # history = efficientnet(training_data, validation_data)
    # plot_results(history)
