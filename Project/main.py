import tensorflow as tf
import os
import matplotlib.pyplot as plt
import pathlib
import shutil
from keras_preprocessing.image import ImageDataGenerator
from pathlib import Path
from random import shuffle
from glob import glob
from keras.layers import BatchNormalization
from tensorflow.python.keras.applications.inception_v3 import InceptionV3

TRAIN_DATASET_PATH = pathlib.Path("../Project/datasets/boy_or_girl/train")
TEST_DATASET_PATH = pathlib.Path("../Project/datasets/boy_or_girl/test")
DATA_DATASET_PATH = pathlib.Path("../Project/datasets/boy_or_girl/data")


# create test set if not exist
def create_test_set():
    # if files in test directory exists, exit function
    if os.path.exists(TEST_DATASET_PATH) and os.listdir(TEST_DATASET_PATH):
        return

    # create test directory if not exist
    Path(TEST_DATASET_PATH).mkdir(parents=True, exist_ok=True)

    # store file paths in lists
    men_test_files = glob(str(TRAIN_DATASET_PATH / "men/*"))
    women_test_files = glob(str(TRAIN_DATASET_PATH / "women/*"))
    test_files = men_test_files + women_test_files

    # shuffles paths in list several times
    for i in range(5):
        shuffle(test_files)

    # copy only 1/5 of the files to test directory
    for i in range(len(test_files) // 5):
        shutil.copy(test_files[i], TEST_DATASET_PATH)


# split dataset to training and validation data, randomises it also a little
def read_dataset():
    data_gen = ImageDataGenerator(rescale=1. / 255,
                                  width_shift_range=0.2,
                                  validation_split=0.1,
                                  height_shift_range=0.2,
                                  shear_range=0.2,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  zoom_range=0.2)

    _training_data = data_gen.flow_from_directory(DATA_DATASET_PATH,
                                                  target_size=(150, 150),
                                                  class_mode='binary',
                                                  batch_size=32,
                                                  subset='training'
                                                  )

    _validation_data = data_gen.flow_from_directory(DATA_DATASET_PATH,
                                                    target_size=(150, 150),
                                                    class_mode='binary',
                                                    batch_size=32,
                                                    subset='validation')

    return _training_data, _validation_data


def inception():
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
    x = BatchNormalization()(x)  # TODO test if it works actually better
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(pre_trained_model.input, x)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer='Adam', metrics=['accuracy'])

    return model.fit_generator(training_data, epochs=2, steps_per_epoch=5, validation_data=validation_data)


def plot_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
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
    # create_test_set()
    training_data, validation_data = read_dataset()

    history = inception()
    plot_results(history)
