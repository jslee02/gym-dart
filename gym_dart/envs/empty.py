import numpy as np
from gym import utils
from gym_dart.envs import dart_env


class DartEmptyEnv(dart_env.DartEnv):
    def __init__(self):
        pass

    def _step(self, action):
        print('DartEmptyEnv._step')

        ob = 0.0
        reward = 0.0
        done = True

        return ob, reward, done, {}

    def _reset(self):
        pass

    def _render(self, mode='human', close=False):
        return

    def _close(self):
        pass

    def _seed(self, seed=None):
        return []
