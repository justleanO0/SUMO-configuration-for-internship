from gym.envs.registration import register

register(
    id='gym_sumo-v0',
    entry_point='gym_sumo.envs.sumo_env:SumoEnv',
)
'''register(
    id='gym_sumo-v1',
    entry_point='gym_sumo.envs.test_ng_env:NgSumoEnv',
)
'''