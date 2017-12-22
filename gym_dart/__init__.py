from gym.envs.registration import register

register(
    id='dart-v0',
    entry_point='gym_dart.envs:DartEnv',
)

register(
    id='dart-extrahard-v0',
    entry_point='gym_dart.envs:DartExtraHardEnv',
)

register(
    id='DartCartPole-v0',
    entry_point='gym_dart.envs:DartCartPoleEnv',
)

register(
    id='DartEmpty-v0',
    entry_point='gym_dart.envs:DartEmptyEnv',
)
