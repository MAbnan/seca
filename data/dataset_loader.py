"""
Dataset Loader for SECA

Loads and preprocesses dataset for training.
"""

import tensorflow as tf


def load_dataset():
    """
    Load MNIST dataset.

    Returns
    -------
    x_train, x_test, y_train, y_test
    """

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # normalize images
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # reshape for CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return x_train, x_test, y_train, y_test