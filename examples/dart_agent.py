import gym
import gym_dart
import numpy as np
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:
        env = gym.make(sys.argv[1])
    else:
        #env = gym.make('DartEmpty-v0')
        env = gym.make('DartCartPole-v0')

    # env.env.disableViewer = False

    env.reset()

    for i in range(1000):
        print('i:', i, ', ', env.step([0, 0]))
        # env.render()

    # env.render(close=True)
