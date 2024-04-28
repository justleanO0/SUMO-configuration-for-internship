# -*- coding: utf-8 -*-
"""
Created on Sun May 22 15:51:50 2022

@author: Jialin
"""

import numpy as np
from tqdm import tqdm
from keras.models import load_model
from dqn import DQNAgent
from policy_gradient import PGAgent
import gym
import gym_sumo
import matplotlib.pyplot as plt
'''
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in  env_dict:
    if 'gym_sumo-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
'''

#DQN
env = gym.make('gym_sumo-v0')
dqn_agent = DQNAgent()
load_path = "DQN_models/global_model/"
network = load_model(load_path + 'wglobal_step9999.h5')
'''
#PG
#env = gym.make('gym_sumo-v0')
pg_agent = PGAgent()
load_path = "PG_models/"
network = load_model(load_path + 'PGmodel.h5')
'''
def test_DQN(agent, env, network, steps_per_epoch=100, epochs=1, threshold=3, nv=150, t=25):

    next_obs = env.reset(gui=True, numVehicles=nv, thr=threshold, ttulcr=t)
    first_epoch = 0
    avg_speed = 0
    avg_speed_his = []

    pred_a = []
    try:
        for epoch in tqdm(range(first_epoch, epochs)):
            for step in range(steps_per_epoch):
                # curr state
                # get action
                state = next_obs.copy()
                action = network.predict(np.expand_dims(state, axis=0))
                action = np.argmax(action)
                pred_a.append(action)
                
                # do step
                next_obs, rewards_info, done, collision = env.step(action)
                rewards_tot, R_comf, R_eff, R_safe = rewards_info

                avg_speed_his.append(avg_speed)
                avg_speed = (avg_speed*(env.curr_step-1) +
                             next_obs[3])/env.curr_step

        print("Number of real lane change:", env.nb_lc)
        print("Number of collisions:", env.total_collision)
        print("Current step:", env.curr_step)
        print("Avg speed:", avg_speed)
        print("Risky time:", env.risky_time)
        print("Blocking time:", env.block_time)

    except KeyboardInterrupt:
        print("Number of real lane change:", env.nb_lc)
        print("Number of collisions:", env.total_collision)
        print("Avg speed:", avg_speed)
        print("Risky time:", env.risky_time)
        print("Blocking time:", env.block_time)

    xl = np.linspace(1, env.curr_step, env.curr_step)
    plt.plot(xl, avg_speed_his, label="average speed")
    plt.title("avg speed")
    plt.show()

    env.close()
    return avg_speed, env.risky_time, env.block_time


test_DQN(dqn_agent,env,network)
#winsound.Beep(500,1500)
