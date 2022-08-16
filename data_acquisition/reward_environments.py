"""
Implements the centered array experiement, where trajectories that stay near the origin are labelled as good.
"""
from dataclasses import dataclass
import numpy as np
import random


@dataclass
class RewardEnvironment:
    """Parent class of all (environment, reward) pairs"""
    data_dirname: str

    def simulate_trajectories_and_returns(self, num_samples, trajectory_length):
        pass

    def save_unprocessed_trajectories_and_returns(self, trajectories, returns, data_dirname=None):
        """Saves trajectories and returns"""
        if data_dirname is None:
            data_dirname = self.data_dirname
        np.savetxt(f'{data_dirname}/n{trajectories.shape[0]}_t{trajectories.shape[1]}_trajectories.txt', trajectories)
        np.savetxt(f'{data_dirname}/n{trajectories.shape[0]}_t{trajectories.shape[1]}_returns.txt', returns)
        print(f'saved trajectories and returns to {data_dirname}')

    def load_unprocessed_trajectories_and_returns(self, num_samples, trajectory_length, data_dirname=None):
        if data_dirname is None:
            data_dirname = self.data_dirname
        trajectories = np.loadtxt(f'{data_dirname}/n{num_samples}_t{trajectory_length}_trajectories.txt')
        returns = np.loadtxt(f'{data_dirname}/n{num_samples}_t{trajectory_length}_returns.txt')

        print(f'trajectories and returns loaded from {data_dirname}')

        return trajectories, returns


@dataclass
class ToyCenteredArray(RewardEnvironment):
    """
    Deterministic array class assigns a positive return to all trajectories that stay in the range [-1, 1].
    It assigns zero return to any other trjajectory.
    """

    R_MAX: int = 1    # max reward
    S_MIN: int = -3   # state space is evenly spaced integers from [S_MIN, S_MAX]
    data_dirname: str = 'data/toy_centered_array'

    LEFT = 0          # for readability, label the actions 'left' and 'right'
    RIGHT = 1

    def optimal_actions(self, s: int):
        """Optimal actions define define the reward"""
        assert (s >= self.S_MIN) and (s <= -1*self.S_MIN)

        if s > 1:
            return [self.LEFT]
        elif s < -1:
            return [self.RIGHT]
        else:
            return [self.LEFT, self.RIGHT]


    def uniform_actions(self, s: int):
        assert (s >= self.S_MIN) and (s <= -1*self.S_MIN)
        if s == self.S_MIN:
            possible_actions = [self.RIGHT]
        elif s == -1*self.S_MIN:
            possible_actions = [self.LEFT]
        else:
            possible_actions = [self.LEFT, self.RIGHT]

        return possible_actions

    def deterministic_transition_kernel(self, s: int, a: int):
        """Deterministic Transition Kernel"""
        if a == self.LEFT:
            return s-1
        else:
            return s+1

    def simulate_trajectories_and_returns(self, num_samples, trajectory_length):

        n_optimal = num_samples // 2
        n_random = num_samples // 2

        trajectories = np.empty((n_optimal + n_random, trajectory_length))
        trajectory_return = np.empty((n_optimal+n_random, 1))

        for i in range(n_optimal):

            good_trajectory = 1
            # simulate a T-length trajectory

            # always start at origin
            s = 0
            for t in range(trajectory_length):
                trajectories[i, t] = s
                a = random.sample(self.optimal_actions(s), 1)[0]
                # always true for optimal actions
                if a in self.optimal_actions(s):
                    good_trajectory = 1
                else:
                    good_trajectory = 0

                s = self.deterministic_transition_kernel(s, a)

            trajectory_return[i] = good_trajectory

        for j in range(n_random):

            good_trajectory = 1

            # simulate a T-length trajectory, starting at the origin
            s = 0
            for t in range(trajectory_length):
                trajectories[n_optimal+j, t] = s
                a = random.sample(self.uniform_actions(s), 1)[0]
                # always true for optimal actions
                if a in self.optimal_actions(s):
                    good_trajectory = 1
                else:
                    good_trajectory = 0

                s = self.deterministic_transition_kernel(s, a)

            trajectory_return[n_optimal+j] = good_trajectory

        return trajectories, trajectory_return
