"""
Model Builder for SECA

This module converts a genome representation into a neural network model.
The architecture is dynamically created based on genome parameters.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


INPUT_SHAPE = (28, 28, 1)
NUM_CLASSES = 10


def build_model(genome):
    """
    Build a neural network model from a genome.

    Parameters
    ----------
    genome : dict
        Genome describing architecture

    Returns
    -------
    model : keras.Model
    """

    model = models.Sequential()

    # Input layer
    model.add(layers.Input(shape=INPUT_SHAPE))
    current_size = INPUT_SHAPE[0]

    # Convolutional stages
    for stage in genome["stages"]:

        filters = stage["filters"]
        kernel = stage["kernel"]
        pool = stage["pool"]

        model.add(
            layers.Conv2D(
                filters=filters,
                kernel_size=(kernel, kernel),
                activation="relu",
                padding="same"
            )
        )

        model.add(layers.BatchNormalization())

        if pool == 2:
            if current_size >= 2:
                model.add(layers.MaxPooling2D((2, 2)))
                current_size //= 2

    # Flatten
    model.add(layers.Flatten())

    # Dense layer
    model.add(
        layers.Dense(
            genome["dense"],
            activation="relu"
        )
    )

    # Dropout
    if genome["dropout"] > 0:
        model.add(
            layers.Dropout(genome["dropout"])
        )

    # Output layer
    model.add(
        layers.Dense(
            NUM_CLASSES,
            activation="softmax"
        )
    )

    # Compile model
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model