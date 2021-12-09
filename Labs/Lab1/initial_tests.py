import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def main():
    iris = sns.load_dataset('iris')
    iris.head()  # powinno zwracac 5 pierwszych linijek z datasetu na ekran
    sns.pairplot(iris, hue='species', height=3)
    plt.scatter(iris['petal_length'], iris['petal_width'])

    model = LinearRegression(fit_intercept=True)
    model.fit(iris[['petal_length']], iris['petal_width'])

    y_predict = model.predict(iris['petal_length'][:, np.newaxis])
    plt.scatter(iris['petal_length'], iris['petal_width'])
    plt.plot(iris['petal_length'], y_predict)
