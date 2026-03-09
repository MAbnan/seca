"""
Trainer Module for SECA

Responsible for:
1. Training neural architectures
2. Evaluating validation accuracy
3. Returning training history
"""

import tensorflow as tf


EPOCHS = 3
BATCH_SIZE = 64


def train_model(model, x_train, y_train, x_test, y_test):
    """
    Train a neural network model.

    Parameters
    ----------
    model : keras.Model
        Neural network created from genome

    x_train : array
    y_train : array

    x_test : array
    y_test : array

    Returns
    -------
    history : keras history object
    val_accuracy : float
    """

    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        verbose=0
    )

    # Extract validation accuracy
    val_accuracy = history.history["val_accuracy"][-1]

    return history, val_accuracy