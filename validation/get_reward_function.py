import numpy as np
import tensorflow as tf
import keras
import sys

sys.path.append('../')
from training.model import get_reward_model
from data_acquisition.main import load_trajectories_and_returns_from_env, process_trajectories


memory_window_length = 2
T = 50
num_windows = T - memory_window_length + 1

NUM_TEST_SAMPLES = 100
# Load and process trajectories
trajectories, returns = load_trajectories_and_returns_from_env('ToyCenteredArray', NUM_TEST_SAMPLES, T)
x_train, y_train = process_trajectories(trajectories, returns, memory_window_length, shuffle=True)


def get_reward_function(return_model_weights_fname, num_windows, memory_window_length):
    """Extracts the reward function from the learned model of return."""

    return_model = get_reward_model(num_windows, memory_window_length)
    return_model.load_weights(return_model_weights_fname)

    reward_model = tf.keras.models.Sequential(return_model.layers[num_windows-2:-1])

    return reward_model, return_model


reward_model, return_model = get_reward_function('../out/model_weights/toy_centered_array/test_500k_sigmoid.h5', num_windows, memory_window_length)


def test_return_vs_reward():
    # Compute the output of return model
    output_of_return_model = return_model.predict([x_train[:, i] for i in range(x_train.shape[1])])
    output_of_return_model = output_of_return_model.reshape(-1, )


# Compute the output of the reward model by summing individual rewards rewards
rewards = reward_model.predict(x_train.reshape(-1, memory_window_length))
rewards = rewards.reshape(-1, num_windows)
predicted_returns = rewards.sum(axis=-1)

# Make sure they are similar
print(predicted_returns - y_train.reshape(-1))
