"""This is a model for classifying fashion MNIST objects with Tensorflow and Keras. It is based on the architecture
of LeNet-5. However, more kernels are used in the convolution layer. ReLu acivations, max-pooling layers and dropout
are used. """

from typing import Any

import numpy as np
from tensorflow.keras.callbacks import TensorBoard  # new!
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,  # new!
                                     MaxPooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

from settings_fashion_mnist_LeNet5 import (BATCH_SIZE, EPOCHS, N_CLASSES, NAME,
                                           OUTPUT_DIR, SAVE_FORMAT)


def load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The function loads the fashion mnist data and assigns it to the four variables for the training and test data of
    the features and labels.
    Returns: All test and training data of features and labels in four variables.
    """
    (X_train, y_train), (X_valid, y_valid) = fashion_mnist.load_data()
    return (X_train, y_train), (X_valid, y_valid)


def preprocess_data(
    X_train: np.ndarray, X_valid: np.ndarray, y_train: np.ndarray, y_valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    The digits are converted from integer values to glue comma numbers in order to scale them to a range from 0 to 1
    and the integer values of the data are converted to the 1-out-of-n format. N_CLASSES is ten, because there are
    10 digits.
    Args:
        X_train: Feature train data from the dataset.
        X_valid: Feature test data from the dataset.
        y_train: Label train data from the dataset.
        y_valid: Lable test data from the dataset.

    Returns:
        A tuple of the four categorised variables of the features and the label.
    """
    X_train = X_train.reshape(60000, 28, 28, 1).astype("float32")
    X_valid = X_valid.reshape(10000, 28, 28, 1).astype("float32")
    X_train /= 255
    X_valid /= 255
    y_train = to_categorical(y_train, N_CLASSES)
    y_valid = to_categorical(y_valid, N_CLASSES)
    return (X_train, X_valid, y_train, y_valid)


def neural_network_architecture(
    X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, y_valid: np.ndarray
) -> Any:
    """
    The function builds, configures and trains the model.
    Args:
        X_train: The categorised feature train data from the dataset.
        X_valid: The categorised feature test data from the dataset.
        y_train: The categorised label train data from the dataset.
        y_valid: The categorised lable test data from the dataset.

    Returns:
        The trained model.
    """
    model = Sequential()

    model.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1))
    )

    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(N_CLASSES, activation="softmax"))

    model.summary()
    # configure model
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    # Set TensorBoard logging directory
    tensorboard = TensorBoard(log_dir="logs/deep-net")
    # fit model
    model.fit(
        X_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(X_valid, y_valid),
        callbacks=[tensorboard],
    )
    return model


def model_save(model: Any) -> None:
    """
    This function saves the model so that it can be reloaded later for an application.
    Args:
        model: the trained model.

    Returns:
        None.
    """
    model.save(f"{OUTPUT_DIR}{NAME}", save_format=SAVE_FORMAT)


def evaluate(model: Any, x_valid: np.ndarray, y_valid: np.ndarray) -> None:
    """
    The function calculates the validation loss and the validation accuracy of the last epoch.
    Args:
        model: The trained model.
        x_valid: The training features.
        y_valid: The test labels

    Returns:
        None
    """
    val_loss, val_acc = model.evaluate(x_valid, y_valid)
    print(f"validation loss: {val_loss}")
    print(f"validation: {val_acc}")


def main():
    print("Наталья, я очень тебя люблю")
    (X_train, y_train), (X_valid, y_valid) = load_data()
    X_train, X_valid, y_train, y_valid = preprocess_data(
        X_train, X_valid, y_train, y_valid
    )
    model = neural_network_architecture(X_train, y_train, X_valid, y_valid)
    model_save(model)
    evaluate(model, X_valid, y_valid)


if __name__ == "__main__":
    main()
