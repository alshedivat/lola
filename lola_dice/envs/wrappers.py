import numpy as np
from collections import deque

from gym import ObservationWrapper


class FrameStack(ObservationWrapper):
    """
    Stacks k last frames. Not the most efficient implementation.
    """
    def __init__(self, env, k):
        super(FrameStack, self).__init__(env)
        self.NUM_STATES = env.NUM_STATES * k
        self.NUM_ACTIONS = env.NUM_ACTIONS
        self.max_steps = env.max_steps
        self.concat = lambda x: np.concatenate(x, axis=-1)
        self.frames = deque([], maxlen=k)
        self.k = k

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(observation)
        observation = list(map(self.concat, zip(*list(self.frames))))
        return observation, info

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        observation = list(map(self.concat, zip(*list(self.frames))))
        return observation, reward, done, info
