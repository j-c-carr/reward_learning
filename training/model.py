import numpy as np
import tensorflow as tf
import keras


def get_reward_model(num_windows: int, memory_window_length: int):
    """Creates simple reward model"""

    inputs = [keras.Input(shape=(memory_window_length,)) for _ in range(num_windows)]

    # Create the same branch for each input
    layer1 = keras.layers.Dense(32, input_shape=inputs[0].shape, activation='relu')
    layer2 = keras.layers.Dense(8, activation='relu')
    layer3 = keras.layers.Dense(1, activation='sigmoid')

    branches = [layer3(layer2(layer1(inputs[i]))) for i in range(len(inputs))]

    final_layer = keras.layers.Lambda(lambda x: keras.backend.sum(x, axis=0, keepdims=True))

    model = keras.Model(inputs=inputs, outputs=final_layer(branches))
    # print(model.output_shape)
    # print(model.summary())

    mse = keras.losses.MeanSquaredError()
    model.compile(optimizer='adam', loss=mse, metrics=['accuracy', 'mean_absolute_error'])

    return model

