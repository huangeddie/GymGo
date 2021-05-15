from gym.envs.registration import register

register(
    id='go-v0',
    entry_point='gym_go.envs:GoEnv',
)
register(
    id='go-extrahard-v0',
    entry_point='gym_go.envs:GoExtraHardEnv',
)
