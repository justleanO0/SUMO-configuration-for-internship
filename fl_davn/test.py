import gym
'''env_dict = gym.envs.registration.registry.env_specs.copy()
for env in  env_dict:
    if 'gym_sumo-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]'''

import gym_sumo
from dqn import DQNAgent
from policy_gradient import PGAgent

env = gym.make('gym_sumo-v0')
agent = DQNAgent()
#agent = PGAgent()
agent.train(env)
