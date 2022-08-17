"""
Loads and processes trajectories for different toy environments
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from data_acquisition.reward_environments import RewardEnvironment, ToyCenteredArray

ENVS = ['ToyCenteredArray']

def load_trajectories_and_returns_from_file(data_dirname, num_samples, trajectory_length):
    """Loads unprocessed trajectories from file in the data folder"""
    env = RewardEnvironment(data_dirname=data_dirname)
    trajectories, returns = env.load_unprocessed_trajectories_and_returns(num_samples, trajectory_length)

    return trajectories, returns


def load_trajectories_and_returns_from_env(env_name, num_samples, trajectory_length):

    assert env_name in ENVS, f'please specify a valid environment among {ENVS}'
    if env_name == 'ToyCenteredArray':
        env = ToyCenteredArray()

    trajectories, returns = env.simulate_trajectories_and_returns(num_samples, trajectory_length)

    return trajectories, returns


def process_trajectories(trajectories, returns, window_shape, shuffle=True):
    """Chops each trajectory into proper input"""

    if shuffle:
        p = np.random.permutation(trajectories.shape[0])
        trajectories = trajectories[p]
        returns = returns[p]

    processed_trajectories = np.array(
        [sliding_window_view(trajectory, window_shape=window_shape) for trajectory in trajectories])
    processed_returns = returns
    return processed_trajectories, processed_returns


if __name__ == '__main__':
    env = ToyCenteredArray()
    trajectories, returns = env.simulate_trajectories_and_returns(500_000, 50)
    env.save_unprocessed_trajectories_and_returns(trajectories, returns)

    # trajectories, returns = load_unprocessed_trajectories_and_returns('data/exp1')
    print(trajectories.shape)
    print(returns.shape)
