import os
from os import path
import numpy as np
import os

from gym import error, spaces
from gym.utils import seeding
import numpy as np
from os import path
import gym
import six

import gym
from gym import error, spaces

try:
    import dartpy as dart
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install dartpy with 'sudo apt-get install "
                                       "python-dartpy.)".format(e))

import logging

logger = logging.getLogger(__name__)


#    When implementing an environment, override the following methods
#    in your subclass:
#        _step
#        _reset
#        _render
#        _close
#        _seed
#    And set the following attributes:
#        action_space: The Space object corresponding to valid actions
#        observation_space: The Space object corresponding to valid observations
#        reward_range: A tuple corresponding to the min and max possible rewards
#    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.

class DartEnv(gym.Env):
    def __init__(self, model_paths, frame_skip, observation_size, action_bounds, \
                 dt=0.002, obs_type="parameter", action_type="continuous", visualize=True, disableViewer=False,\
                 screen_width=80, screen_height=45):
        assert obs_type in ('parameter', 'image')
        assert action_type in ("continuous", "discrete")

        self.viewer = None

        if len(model_paths) < 1:
            raise ValueError("At least one model file is needed.")

        if isinstance(model_paths, str):
            model_paths = [model_paths]

        # Convert everything to full-path
        full_paths = []
        for model_path in model_paths:
            if model_path.startswith("/"):
                full_path = model_path
            else:
                full_path = os.path.join(os.path.dirname(__file__), "assets", model_path)
            if not path.exists(full_path):
                raise IOError("File %s does not exist" % full_path)
            full_paths.append(full_path)

        self.dart_world = dart.simulation.World()

        urdf_loader = dart.utils.DartLoader()

        for full_path in full_paths:
            if full_path[-5:] == '.urdf':
                skeleton = urdf_loader.parseSkeleton(full_path)
                self.dart_world.addSkeleton(skeleton)
            else:
                raise NotImplementedError

        # Assume that the skeleton of interest is always the last one
        self.robot_skeleton = self.dart_world.getSkeleton(self.dart_world.getNumSkeletons() - 1)

        # for joint_index in range(self.robot_skeleton.getNumJoints()):
        #     joint = self.robot_skeleton.getJoint(joint_index)
        #     for dof_index in range(joint.getNumDofs()):
        #         dof = joint.getDof(dof_index)
        #         if dof.hasPositionLimit(dof_index):
        #             joint.setPositionLimitEnforced(True)
        #             break

        self._obs_type = obs_type
        self.frame_skip = frame_skip
        self.visualize = visualize  # Show the window or not
        self.disableViewer = disableViewer

        # Random perturbation
        self.add_perturbation = False
        self.perturbation_parameters = [0.05, 5, 2]  # probability, magnitude, bodyid, duration
        self.perturbation_duration = 40
        self.perturb_force = np.array([0, 0, 0])

        # Assert not done
        self.obs_dim = observation_size
        self.act_dim = len(action_bounds[0])

        # For discrete instances, action_space should be defined in the subclass
        if action_type == "continuous":
            self.action_space = spaces.Box(action_bounds[1], action_bounds[0])

        self.track_skeleton_id = -1  # track the last skeleton's com by default

        # Initialize the viewer, get the window size
        # Initial here instead of in _render
        # in image learning
        self.screen_width = screen_width
        self.screen_height = screen_height
        # self._get_viewer()

        # Give different observation space for different kind of envs
        if self._obs_type == 'parameter':
            high = np.inf * np.ones(self.obs_dim)
            low = -high
            self.observation_space = spaces.Box(low, high)
        elif self._obs_type == 'image':
            # Change to grayscale image later
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height))
        else:
            raise error.Error('Unrecognized observation type: {}'.format(self._obs_type))

        self._seed()

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------
    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized and after every reset
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def _reset(self):
        self.perturbation_duration = 0
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.robot_skeleton.getNumDofs(), 1) and qvel.shape == (self.robot_skeleton.getNumDofs(), 1)
        self.robot_skeleton.setPositions(qpos)
        self.robot_skeleton.setVelocities(qvel)

    def set_state_vector(self, state):
        self.robot_skeleton.setPositions(state[0:int(len(state)/2)])
        self.robot_skeleton.setVelocities(state[int(len(state)/2):])

    @property
    def dt(self):
        return self.dart_world.getTimeStep() * self.frame_skip

    def do_simulation(self, tau, n_frames):
        if self.add_perturbation:
            if self.perturbation_duration == 0:
                self.perturb_force *= 0
                if np.random.random() < self.perturbation_parameters[0]:
                    axis_rand = np.random.randint(0, 2, 1)[0]
                    direction_rand = np.random.randint(0, 2, 1)[0] * 2 - 1
                    self.perturb_force[axis_rand] = direction_rand * self.perturbation_parameters[1]

            else:
                self.perturbation_duration -= 1

        for _ in range(n_frames):
            if self.add_perturbation:
                self.robot_skeleton.getBodyNode(self.perturbation_parameters[2]).addExtForce(self.perturb_force)

            self.robot_skeleton.setForces(tau)
            self.dart_world.step()

    def _step(self, action):
        raise NotImplementedError

    def _render(self, mode='human', close=False):
        # TODO(JS)
        return
