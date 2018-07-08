"""
Coin Game environment.
"""
import gym
import numpy as np

from gym.spaces import Discrete, Tuple
from gym.spaces import prng


class CoinGameVec:
    """
    Vectorized Coin Game environment.
    Note: slightly deviates from the Gym API.
    """
    NUM_AGENTS = 2
    NUM_ACTIONS = 4
    MOVES = [
        np.array([0,  1]),
        np.array([0, -1]),
        np.array([1,  0]),
        np.array([-1, 0]),
    ]

    def __init__(self, max_steps, batch_size, grid_size=3):
        self.max_steps = max_steps
        self.grid_size = grid_size
        self.batch_size = batch_size
        # The 4 channels stand for 2 players and 2 coin positions
        self.ob_space_shape = [4, grid_size, grid_size]

        self.step_count = None

    def reset(self):
        self.step_count = 0
        self.red_coin = prng.np_random.randint(2, size=self.batch_size)
        # Agent and coin positions
        self.red_pos  = prng.np_random.randint(
            self.grid_size, size=(self.batch_size, 2))
        self.blue_pos = prng.np_random.randint(
            self.grid_size, size=(self.batch_size, 2))
        self.coin_pos = np.zeros((self.batch_size, 2), dtype=np.int8)
        for i in range(self.batch_size):
            # Make sure coins don't overlap
            while self._same_pos(self.red_pos[i], self.blue_pos[i]):
                self.blue_pos[i] = prng.np_random.randint(self.grid_size, size=2)
            self._generate_coin(i)
        return self._generate_state()

    def _generate_coin(self, i):
        self.red_coin[i] = 1 - self.red_coin[i]
        # Make sure coin has a different position than the agents
        success = 0
        while success < 2:
            self.coin_pos[i] = prng.np_random.randint(self.grid_size, size=(2))
            success  = 1 - self._same_pos(self.red_pos[i],
                                          self.coin_pos[i])
            success += 1 - self._same_pos(self.blue_pos[i],
                                          self.coin_pos[i])

    def _same_pos(self, x, y):
        return (x == y).all()

    def _generate_state(self):
        state = np.zeros([self.batch_size] + self.ob_space_shape)
        for i in range(self.batch_size):
            state[i, 0, self.red_pos[i][0], self.red_pos[i][1]] = 1
            state[i, 1, self.blue_pos[i][0], self.blue_pos[i][1]] = 1
            if self.red_coin[i]:
                state[i, 2, self.coin_pos[i][0], self.coin_pos[i][1]] = 1
            else:
                state[i, 3, self.coin_pos[i][0], self.coin_pos[i][1]] = 1
        return state

    def step(self, actions):
        for j in range(self.batch_size):
            ac0, ac1 = actions[j]
            assert ac0 in {0, 1, 2, 3} and ac1 in {0, 1, 2, 3}

            # Move players
            self.red_pos[j] = \
                (self.red_pos[j] + self.MOVES[ac0]) % self.grid_size
            self.blue_pos[j] = \
                (self.blue_pos[j] + self.MOVES[ac1]) % self.grid_size

        # Compute rewards
        reward_red, reward_blue = [], []
        for i in range(self.batch_size):
            generate = False
            if self.red_coin[i]:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red.append(1)
                    reward_blue.append(0)
                elif self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red.append(-2)
                    reward_blue.append(1)
                else:
                    reward_red.append(0)
                    reward_blue.append(0)

            else:
                if self._same_pos(self.red_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red.append(1)
                    reward_blue.append(-2)
                elif self._same_pos(self.blue_pos[i], self.coin_pos[i]):
                    generate = True
                    reward_red.append(0)
                    reward_blue.append(1)
                else:
                    reward_red.append(0)
                    reward_blue.append(0)

            if generate:
                self._generate_coin(i)

        reward = [np.array(reward_red), np.array(reward_blue)]
        self.step_count += 1
        done = np.array([
            (self.step_count == self.max_steps) for _ in range(self.batch_size)
        ])
        state = self._generate_state()

        return state, reward, done
