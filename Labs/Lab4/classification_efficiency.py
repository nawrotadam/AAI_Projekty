from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt

train_generator1 = ImageDataGenerator(
    validation_split=0.2,
    rescale=1. / 255,
    horizontal_flip=True
)

train_generator2 = ImageDataGenerator(
    validation_split=0.2,
    rescale=1. / 255,
    brightness_range=[0.5, 1.0]
)

train_generator3 = ImageDataGenerator(
    validation_split=0.2,
    rescale=1. / 255,
    zoom_range=[0.5, 1]
)

train_generators = [train_generator1, train_generator2, train_generator3]


def main():
    # !wget - -no - check - certificate \
    #         https: // storage.googleapis.com / mledu - datasets / cats_and_dogs_filtered.zip \
    #                   - O / tmp / cats_and_dogs_filtered.zip
    data_dir = "../Labs/Lab4/cats_and_dogs_filtered/train"
    validation_dir = "../Labs/Lab4/cats_and_dogs_filtered/validation"

    for train_generator in train_generators:
        validation_generator = ImageDataGenerator(
            validation_split=0.2,
            rescale=1. / 255)

        train_generator = train_generator.flow_from_directory(
            data_dir,
            target_size=(130, 130),
            subset='training',
            seed=69)

        validation_generator = validation_generator.flow_from_directory(
            validation_dir,
            target_size=(130, 130),
            subset='validation',
            seed=69)

        model = Sequential([
          layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(130, 130, 3)),
          layers.MaxPooling2D(),
          layers.Conv2D(32, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Conv2D(64, 3, padding='same', activation='relu'),
          layers.MaxPooling2D(),
          layers.Flatten(),
          layers.Dense(256, activation='relu'),
          layers.Dense(2, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model.summary()

        history = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=10
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(10)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()
