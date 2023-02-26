from nnet.model import Sequential
from nnet.layers import Input, Dense, Dropout, Conv2D, MaxPool2D, Flatten, Reshape
from nnet.utils import one_hot
import numpy as np
import pandas as pd

fashion = pd.read_csv(
    "/media/hari31416/Hari_SSD/Users/harik/Desktop/APL745/Assignments/Assignment_4/data/fashion-mnist_train.csv"
)


def check_params():

    X = fashion.iloc[:, 1:].values
    y = fashion.iloc[:, 0].values
    y_oh = one_hot(y, 10)

    X = X.T
    X = X / X.max()
    nx, m = X.shape
    ny, m = y_oh.shape
    print(m, ny, nx)

    model = Sequential("my_model")
    # inp = Input((28, 28, 3), name="First Input")
    inp = Input((nx,), name="First Input")

    # dense = Dense(20, activation="relu", name="Dense 1")
    # model.add(inp)
    # model.add(dense)
    # model.add(Dense(10, name="Dense A", activation="relu"))
    # # model.add(Dropout(0.2, "dropout 1"))
    # model.add(Dense(8, name="Dense C", activation="relu"))
    # model.add(Dense(ny, name="Dense Z", activation="softmax"))
    model.add(inp)
    model.add(Reshape((28, 28, 1), name="Reshape"))
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu", name="Conv 1"))
    model.add(MaxPool2D((2, 2), name="MaxPool 1"))
    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu", name="Conv 2"))
    model.add(Flatten(name="Flatten"))
    # model.add(Dense(128, name="Dense 1", activation="relu"))
    model.add(Dense(10, name="Output", activation="softmax"))

    model.summary()

    model.compile(
        loss="categorical_cross_entropy",
        metrics=["accuracy", "precision", "recall", "f1"],
        initializer="glorot",
    )


def check_convolve():

    X = fashion.iloc[:, 1:].values
    y = fashion.iloc[:, 0].values
    y_oh = one_hot(y, 10)

    X = X.T
    X = X / X.max()
    nx, m = X.shape
    ny, m = y_oh.shape
    print(m, ny, nx)

    model = Sequential("my_model")
    # inp = Input((28, 28, 3), name="First Input")
    inp = Input((nx,), name="First Input")

    model.add(inp)
    model.add(Reshape((28, 28, 1), name="Reshape"))
    model.add(Conv2D(32, (3, 3), padding="same", activation="relu", name="Conv 1"))
    model.add(MaxPool2D((2, 2), name="MaxPool 1"))
    model.add(Conv2D(64, (3, 3), padding="valid", activation="relu", name="Conv 2"))
    model.add(MaxPool2D((2, 2), name="MaxPool 2"))
    model.add(Flatten(name="Flatten"))
    # model.add(Dense(128, name="Dense 1", activation="relu"))
    model.add(Dense(10, name="Output", activation="softmax"))

    model.summary()

    model.compile(
        loss="categorical_cross_entropy",
        metrics=["accuracy"],
        initializer="glorot",
    )

    # x = X[:, :16]
    x = X[:, :64]
    for l in model.layers:
        x = l.forward(x)
        print(f"After Layer: {l}, Shape: {x.shape}")
    # print(x.shape)
    # x = model[0].forward(x)
    # print(f"After shape: {x.shape}, Layer: {model[0]}")
    # x = model[1].forward(x)
    # print(f"After shape: {x.shape}, Layer: {model[1]}")
    # x = model[2].forward(x)
    # print(f"After shape: {x.shape}, Layer: {model[2]}")
    # x = model[3].forward(x)
    # print(f"After shape: {x.shape}, Layer: {model[3]}")
    # x = model[4].forward(x)
    # print(f"After shape: {x.shape}, Layer: {model[4]}")
    # x = model[5].forward(x)
    # print(f"After shape: {x.shape}, Layer: {model[5]}")
    # x = model[6].forward(x)
    # print(f"After shape: {x.shape}, Layer: {model[6]}")
    # x = model[7].forward(x)
    # print(f"After shape: {x.shape}, Layer: {model[7]}")


if __name__ == "__main__":
    check_convolve()
