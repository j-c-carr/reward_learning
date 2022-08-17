"""
Estimates the fundamental limitation of a reward model using Hoeffding's inequality
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('~/.config/matplotlib/stylelib/highres.mplstyle')

from get_reward_function import get_reward_function

import sys
sys.path.append('../')
from data_acquisition.main import load_trajectories_and_returns_from_env, process_trajectories


def plot_hoeffding_limits(GMIN=0, GMAX=1):
    """Plots the limits of hoeffding's inequality"""

    # eps is error tolerance
    # prob_lower_bound = 1 - 2*np.exp(-2*NUM_TEST_SAMPLES*np.power(eps, 2))
    # Express :eps: as a function of :num_test_samples: for a given confidence interval
    sample_complexity_error = lambda m, conf: np.sqrt(((GMAX-GMIN)**2 * -np.log((1-conf)/2)) / (2*m))
    sample_sizes = np.arange(5_000, 500_000, 100)

    for c in [0.9, 0.99, 0.999, 0.9999]:
        error_tolerance = [sample_complexity_error(m, c) for m in sample_sizes]
        plt.plot(sample_sizes, error_tolerance, label=f'c={c}')

    plt.title("Limits of Hoeffding's Bound")
    plt.ylabel("Max. Difference Between Empirical and True Return Error")
    plt.xlabel('Number of Test Samples')
    plt.legend(title='Confidence')
    plt.savefig('../out/hoeffding_test.png', dpi=400)


# Get reward model
memory_window_length = 2
T = 50
num_windows = T - memory_window_length + 1
GMIN = 0
GMAX = 1

plot_hoeffding_limits()
exit()

reward_model, _ = get_reward_function('../out/model_weights/toy_centered_array/test_500k_sigmoid.h5', num_windows, memory_window_length)


# Generate test samples
NUM_TEST_SAMPLES = 125_000
trajectories, returns = load_trajectories_and_returns_from_env('ToyCenteredArray', NUM_TEST_SAMPLES, T)
x_test, y_test = process_trajectories(trajectories, returns, memory_window_length, shuffle=True)

# Compute the error in predicted returns
rewards = reward_model.predict(x_test.reshape(-1, memory_window_length))
rewards = rewards.reshape(-1, num_windows)
returns = np.clip(rewards.sum(axis=-1), GMIN, GMAX)

error_in_return = np.abs(returns - y_test)
empirical_average_error = error_in_return.mean()

# Apply Hoeffding's bound to calculate the prob that expected error in return is within :eps: of the empirical error
eps = 1e-2
prob_lower_bound = 1 - 2*np.exp(-2*NUM_TEST_SAMPLES*np.power(eps, 2))
print(f'Prob. that expected error in return is between {empirical_average_error-eps} and {empirical_average_error+eps} is {prob_lower_bound}')



