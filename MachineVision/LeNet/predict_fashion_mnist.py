"""The programme is used to make a prediction of what the object is from a given fashion mnis objket. Since no object
could be found to test, the programme itself randomly selects a pair from the test data and checks whether the
prediction is correct. """

from typing import Any, Tuple

import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import load_model

from settings_fashion_mnist_LeNet5 import NAME, OUTPUT_DIR
from settings_predict_fashion_mnist import DESCRIBTION_FASHION_MNIST_LABELS


def select_data() -> tuple[np.ndarray, np.ndarray]:
    """
    The function randomly searches for a data pair from the fashion mnist dataset and returns it as a tuple.
    Returns:
        returns the feature and the corresponding label.
    """
    (X_train, y_train), (X_valid, y_valid) = fashion_mnist.load_data()
    rng = default_rng()
    rand_fashion = rng.integers(0, len(X_valid) - 1)
    return (X_valid[rand_fashion], y_valid[rand_fashion])


def shape_formatter(object: np.ndarray) -> np.ndarray:
    """
    Converts a mnist object into the format in which it can be used in the prediction function.
    Args:
        object: mnist fashion object.

    Returns:
        mnist fashion object that can be used in the prediction.
    """
    formatted_object = object.reshape(1, 28, 28)
    return formatted_object


def prediction(model: Any, formatted_object: np.ndarray) -> np.ndarray:
    """
    Makes the prediction of what kind of garment is a fashion_mnist object.
    Args:
        model: Trained loaded model.
        formatted_object: correctly formatted fashion mnis object for the prediction.

    Returns:
        Returns a numpy array with the prediction.
    """
    prd = model.predict(formatted_object)
    return prd


def evaluate(pred: np.ndarray) -> tuple[str, int]:
    """
    Selects the index from the numpy array of the prediction and assigns it a label from the dresses of the fashion
    mnist dresses.
    Args:
        pred: Prediction from the prediction function.

    Returns:
        Assigned index and description of the fashion mnist object from the prediction.
    """
    describtion_fashion_mnist_labels = DESCRIBTION_FASHION_MNIST_LABELS
    # print(pred[0])
    pred = pred[0]
    index = np.where(pred == 1.0)[0][0]
    describtion = describtion_fashion_mnist_labels.get(index)
    return describtion, index


def plot_fashion(obj: np.ndarray) -> None:
    """
    Draws a fashion mnis object.
    Args:
        obj: Object to be drawn.

    Returns:
        None.
    """
    plt.imshow(obj[0], cmap="Greys")
    plt.show()


def check_result(real_result: int, predicted_result: int) -> None:
    """
    Checks if the predicted result is the same as the real result and writes a short answer if the two are the same.
    Args:
        real_result: Real result from the dataset.
        predicted_result: Predicted result from the function.

    Returns:
        None.
    """
    if real_result == predicted_result:
        print(
            f"The prediction is correct!\nPredicted was {predicted_result} and in fact it was {real_result}"
        )
    else:
        print(
            f"The prediction is not correct!\nPredicted was {predicted_result} and in fact it was {real_result}"
        )


def main() -> None:
    model = load_model(f"{OUTPUT_DIR}{NAME}")
    x_valid, y_valid = select_data()
    x_valid_formatted = shape_formatter(x_valid)
    pred = prediction(model, x_valid_formatted)
    describtion, index = evaluate(pred)
    print(f"It's a {describtion}")
    plot_fashion(x_valid_formatted)
    check_result(y_valid, index)


if __name__ == "__main__":
    main()
