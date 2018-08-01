"""
Matching pennies environment.
"""
import gym
import numpy as np

from gym.spaces import Discrete, Tuple

from .common import OneHot


class IteratedMatchingPennies(gym.Env):
    """
    A two-agent vectorized environment for the Matching Pennies game.
    """
    NAME = 'IMP'
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 5

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.payout_mat = np.array([[1, -1],[-1, 1]])
        self.action_space = \
            Tuple([Discrete(self.NUM_ACTIONS), Discrete(self.NUM_ACTIONS)])
        self.observation_space = \
            Tuple([OneHot(self.NUM_STATES), OneHot(self.NUM_STATES)])

        self.step_count = None

    def reset(self):
        self.step_count = 0
        init_state = np.zeros(self.NUM_STATES)
        init_state[-1] = 1
        observations = [init_state, init_state]
        return observations

    def step(self, action):
        ac0, ac1 = action

        self.step_count += 1

        rewards = [self.payout_mat[ac1][ac0], -self.payout_mat[ac1][ac0]]

        state = np.zeros(self.NUM_STATES)
        state[ac0 * 2 + ac1] = 1
        observations = [state, state]

        done = (self.step_count == self.max_steps)

        return observations, rewards, done
