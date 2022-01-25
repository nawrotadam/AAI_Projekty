import numpy as np
import tensorflow as tf
import pathlib
from sklearn.model_selection import train_test_split

DATASET_PATH = pathlib.Path("../Project/datasets/car_dataset")


def read_dataset(dataset_path):
    data = []
    labels = []

    for i in dataset_path.glob("*"):
        image = tf.keras.preprocessing.image.load_img(i, color_mode='rgb', target_size=(280, 280))
        image = np.array(image)
        data.append(image)
        labels.append(i)

    data = np.array(data)
    labels = np.array(labels)

    return train_test_split(data, labels, test_size=0.2, random_state=42)


if __name__ == "__main__":
    X_train, X_test, ytrain, ytest = read_dataset(DATASET_PATH)





