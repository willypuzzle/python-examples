import libs.models as m
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor


def example1(learning_rate=0.01, epochs=10, batch_size=12):
    my_feature = ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    my_label = ([5.0, 8.8, 9.6, 14.2, 18.8, 19.5, 21.4, 26.8, 28.9, 32.0, 33.8, 38.2])

    my_model = m.build_model(learning_rate)

    trained_weight, trained_bias, epochs, rmse = m.train_model(my_model, my_feature,
                                                               my_label, epochs,
                                                               batch_size)
    m.plot_the_model(trained_weight, trained_bias, my_feature, my_label)
    m.plot_the_loss_curve(epochs, rmse)


def example2():
    data_root = "http://github.com/ageron/data/raw/main/"
    lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
    X = lifesat[["GDP per capita (USD)"]].values
    y = lifesat[["Life satisfaction"]].values

    lifesat.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
    plt.axis([23_500, 62_500, 4, 9])
    plt.show()

    model = LinearRegression()
    model.fit(X, y)
    x_new = [[37_655.2]]
    print(model.predict(x_new))

    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(X, y)
    x_new = [[37_655.2]]
    print(model.predict(x_new))
