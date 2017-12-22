import numpy as np
from gym import utils
from gym_dart.envs import dart_env


class DartCartPoleEnv(dart_env.DartEnv):
    def __init__(self):
        control_bounds = np.array([[1.0], [-1.0]])
        self.action_scale = 100
        dart_env.DartEnv.__init__(self, 'cartpole.urdf', 2, 4, control_bounds, dt=0.02, disableViewer=False)
        utils.EzPickle.__init__(self)

    def _step(self, action):
        reward = 1.0

        tau = np.zeros(self.robot_skeleton.getNumDofs())
        tau[0] = action[0] * self.action_scale

        self.do_simulation(tau, self.frame_skip)
        ob = self._get_obs()

        not_done = np.isfinite(ob).all() and (np.abs(ob[1]) <= .2)
        done = not not_done
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([self.robot_skeleton.getPositions(), self.robot_skeleton.getVelocities()]).ravel()

    def reset_model(self):
        self.dart_world.reset()
        qpos = self.robot_skeleton.getPositions() + self.np_random.uniform(low=-.01, high=.01, size=[self.robot_skeleton.getNumDofs(), 1])
        qvel = self.robot_skeleton.getVelocities() + self.np_random.uniform(low=-.01, high=.01, size=[self.robot_skeleton.getNumDofs(), 1])
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        pass
        # self._get_viewer().scene.tb.trans[2] = -3.5
        # self._get_viewer().scene.tb._set_theta(0)
        # self.track_skeleton_id = 0
