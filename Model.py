from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout


def buildModel():
    model = Sequential()
    model.add(
        Convolution2D(filters=32, kernel_size=(3, 3), input_shape=(96, 96, 1), activation='relu')
    )
    model.add(
        MaxPooling2D(pool_size=(2, 2))
    )
    model.add(
        Dropout(rate=0.1)
    )
    model.add(
        Convolution2D(filters=64, kernel_size=(2, 2), activation='relu')
    )
    model.add(
        MaxPooling2D(pool_size=(2, 2))
    )
    model.add(
        Dropout(rate=0.2)
    )
    model.add(
        Convolution2D(filters=128, kernel_size=(2, 2), activation='relu')
    )
    model.add(
        MaxPooling2D(pool_size=(2, 2))
    )
    model.add(
        Dropout(rate=0.3)
    )
    model.add(
        Flatten()
    )
    model.add(
        Dense(units=500, activation='relu')
    )
    model.add(
        Dropout(rate=0.5)
    )
    model.add(
        Dense(units=500, activation='relu')
    )
    model.add(
        Dense(units=30)
    )
    model.compile(optimizer='adam', loss='MSE')
    return model


if __name__ == "__main__":
    model = buildModel()
    model.summary()
