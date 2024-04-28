# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 12:51:35 2023

@author: Jialin

"""

# training with 4 clients

import gym
import matplotlib.pyplot as plt
from dqn import DQNAgent
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
from replayMemory import ReplayMemory
from tqdm import tqdm
import traci
#from reward_show import sum_rewards,avg_100_reward

env_dict = gym.envs.registration.registry.env_specs.copy()
for env in  env_dict:
    if 'gym_sumo-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
import gym_sumo

from scipy import spatial
def cosine_distance(weight1,weight2):
    
    cos_sim1 = 1 - spatial.distance.cosine(np.array(weight1[0]).reshape(weight1[0].size,), np.array(weight2[0]).reshape(weight2[0].size,))
    cos_sim2 = 1 - spatial.distance.cosine(np.array(weight1[2]).reshape(weight1[2].size,), np.array(weight2[2]).reshape(weight2[2].size,))
    cos_sim3 = 1 - spatial.distance.cosine(np.array(weight1[4]).reshape(weight1[4].size,), np.array(weight2[4]).reshape(weight2[4].size,))
    cos_sim4 = 1 - spatial.distance.cosine(np.array(weight1[8]).reshape(weight1[8].size,), np.array(weight2[8]).reshape(weight2[8].size,))

    c = cos_sim1+cos_sim2+cos_sim3+cos_sim4

    return c/4


UPDATE_FREQ = 4 # target network update frequency
MODEL_SAVE_FREQ = 1000
GLOBAL_UPDATE_FREQ = 10
REPLAY_MEMORY_START_SIZE = 33
steps_per_epoch=2000
epochs=1000000
NB_AGENT = 4

w_c = 1
w_r = 1
save_path = "DQN_models/"
total_energy_drone = 0
tot_e_m=0
tot_e_comm = 0
tot_e_comp=0

total_delay_agent1 = 0
total_delay_agent2 = 0
total_delay_agent3 = 0
total_delay_agent4 = 0

agent1 = DQNAgent(ind=1)
agent2 = DQNAgent(ind=2)
agent3 = DQNAgent(ind=3)
agent4 = DQNAgent(ind=4)
global_model = DQNAgent(ind=0)
env = gym.make('gym_sumo-v0',nb_agent=NB_AGENT)

agent1.main_network.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
agent2.main_network.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
agent3.main_network.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
agent4.main_network.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
global_model.main_network.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')


env.start(gui=False, numVehicles=150)
next_obs = env.get_state()

state1 = next_obs[0][0].copy()
state2 = next_obs[1][0].copy()
state3 = next_obs[2][0].copy()
state4 = next_obs[3][0].copy()


my_replay_memory1 = ReplayMemory()
avg_reward1 = 0
avg_speed1 = 0
avg_R_comf1 = 0
avg_R_eff1 = 0
avg_R_safe1 = 0
tot_rewards1 = []
avg_reward_his1 = []
avg_speed_his1 = []
R_comf1_his=[]
R_eff1_his=[]
R_safe1_his=[]
avg_R_comf1_his=[]
avg_R_eff1_his=[]
avg_R_safe1_his=[]
# Metrics
loss_avg1 = keras.metrics.Mean()
loss_avg1_his = []
#tot_rewards1.append(rewards_tot1)
#avg_reward1 = (avg_reward1*(env.curr_step-1) + rewards_info1[0])/env.total_step
avg_reward_his1.append(avg_reward1)



my_replay_memory2 = ReplayMemory()
avg_reward2 = 0
avg_speed2 = 0
avg_R_comf2 = 0
avg_R_eff2 = 0
avg_R_safe2 = 0
tot_rewards2 = []
avg_reward_his2 = []
avg_speed_his2 = []
R_comf2_his=[]
R_eff2_his=[]
R_safe2_his=[]
avg_R_comf2_his=[]
avg_R_eff2_his=[]
avg_R_safe2_his=[]
# Metrics
loss_avg2 = keras.metrics.Mean()
loss_avg2_his = []



my_replay_memory3 = ReplayMemory()
avg_reward3 = 0
avg_speed3 = 0
avg_R_comf3 = 0
avg_R_eff3 = 0
avg_R_safe3 = 0
tot_rewards3 = []
avg_reward_his3 = []
avg_speed_his3 = []
R_comf3_his=[]
R_eff3_his=[]
R_safe3_his=[]
avg_R_comf3_his=[]
avg_R_eff3_his=[]
avg_R_safe3_his=[]
# Metrics
loss_avg3 = keras.metrics.Mean()
loss_avg3_his = []



my_replay_memory4 = ReplayMemory()
avg_reward4 = 0
avg_R_comf4 = 0
avg_R_eff4 = 0
avg_R_safe4 = 0
avg_speed4 = 0
tot_rewards4 = []
avg_reward_his4 = []
avg_speed_his4 = []
R_comf4_his = []
R_eff4_his = []
R_safe4_his = []
avg_R_comf4_his=[]
avg_R_eff4_his=[]
avg_R_safe4_his=[]
# Metrics
loss_avg4 = keras.metrics.Mean()
loss_avg4_his = []



# energy consumption
e=0
avg_energy=0
avg_energy = (avg_energy*(env.curr_step-1) + e)
avg_energy_his = []
avg_energy_his.append(avg_energy)
energy_his = []
energy_his.append(e)
e_m=0
e_comp=0
e_comm=0


# delay
d1,d2,d3,d4=0,0,0,0
avg_delay1 = 0
avg_delay2 = 0
avg_delay3 = 0
avg_delay4 = 0

avg_delay1 = (avg_delay1*(env.curr_step-1) + d1)
avg_delay2 = (avg_delay2*(env.curr_step-1) + d2)
avg_delay3 = (avg_delay3*(env.curr_step-1) + d3)
avg_delay4 = (avg_delay4*(env.curr_step-1) + d4)

avg_delay_his1 = []
avg_delay_his1.append(avg_delay1)
delay_his1 = []
delay_his1.append(d1)

avg_delay_his2 = []
avg_delay_his2.append(avg_delay2)
delay_his2 = []
delay_his2.append(d2)

avg_delay_his3 = []
avg_delay_his3.append(avg_delay3)
delay_his3 = []
delay_his3.append(d3)

avg_delay_his4 = []
avg_delay_his4.append(avg_delay4)
delay_his4 = []
delay_his4.append(d4)


# weights
r_local1_his = []
r_local2_his = []
r_local3_his = []
r_local4_his = []

c_local1_his = []
c_local2_his = []
c_local3_his = []
c_local4_his = []

w_local1_his = []
w_local2_his = []
w_local3_his = []
w_local4_his = []

from keras.models import load_model
network = load_model("DQN_models/global_model/wglobal_init.h5")
g_weights = network.get_weights()
global_model.target_network.set_weights(g_weights)
global_model.main_network.set_weights(g_weights)

model_size = 0
for item in g_weights:
    model_size += item.size
model_bit = model_size*32 # model size in bit
comm_round = 0
true_step = 0
total_step =0
step_his = []

sumr1_his = []
sumr2_his = []
sumr3_his = []
sumr4_his = []

try:
    for epoch in tqdm(range(epochs)):
        if total_step>=32000:
            break
        
        true_step = 0
        
        sumr1 = sumr2 = sumr3 = sumr4 = 0
        #while true_step<steps_per_epoch:
        while true_step<steps_per_epoch:
            true_step += 1
            total_step += 1
                    
            # agents move
            #state1 = next_obs1.copy()
            action1 = agent1.act(state1, agent1.main_network)
            
            #state2 = next_obs2.copy()
            action2 = agent2.act(state2, agent2.main_network)
            
            #state3 = next_obs3.copy()
            action3 = agent3.act(state3, agent3.main_network)
    
            #state4 = next_obs4.copy()
            action4 = agent4.act(state4, agent4.main_network)
    
            step_ = env.step([action1,action2,action3,action4])
            next_obs1, rewards_info1, done1, collision1 = step_[0]
            next_obs2, rewards_info2, done2, collision2 = step_[1]
            next_obs3, rewards_info3, done3, collision3 = step_[2]
            next_obs4, rewards_info4, done4, collision4 = step_[3]
            
            state1 = next_obs1
            state2 = next_obs2
            state3 = next_obs3
            state4 = next_obs4
            
            rewards_tot1, R_comf1, R_eff1, R_safe1 = rewards_info1
            rewards_tot2, R_comf2, R_eff2, R_safe2 = rewards_info2
            rewards_tot3, R_comf3, R_eff3, R_safe3 = rewards_info3
            rewards_tot4, R_comf4, R_eff4, R_safe4 = rewards_info4
            
            #loss_avg1 = keras.metrics.Mean()
            tot_rewards1.append(rewards_tot1)
            avg_reward1 = (avg_reward1*(env.curr_step-1) + rewards_info1[0])/env.total_step
            avg_reward_his1.append(avg_reward1)
            
            # Add experience
            my_replay_memory1.add_experience(action=action1,
                                            frame=next_obs1,
                                            reward=rewards_tot1,
                                            terminal=done1)
            R_comf1_his.append(R_comf1)
            R_eff1_his.append(R_eff1)
            R_safe1_his.append(R_safe1)
            
            avg_R_comf1 = (avg_R_comf1*(env.curr_step-1) + rewards_info1[1])/env.total_step
            avg_R_comf1_his.append(avg_R_comf1)
            avg_R_eff1 = (avg_R_eff1*(env.curr_step-1) + rewards_info1[2])/env.total_step
            avg_R_eff1_his.append(avg_R_eff1)
            avg_R_safe1 = (avg_R_safe1*(env.curr_step-1) + rewards_info1[3])/env.total_step
            avg_R_safe1_his.append(avg_R_safe1)
            
            if agent1.steps_done > REPLAY_MEMORY_START_SIZE:
                loss_value1 = agent1.train_step_(my_replay_memory1)
                loss_avg1.update_state(loss_value1)
                agent1.update_network()
                
            else:
                loss_avg1.update_state(-1)
                
            loss_avg1_his.append(loss_avg1.result().numpy())
            # Copy network from main to target every NETW_UPDATE_FREQ
            if agent1.steps_done % UPDATE_FREQ == 0 and agent1.steps_done > REPLAY_MEMORY_START_SIZE:
                agent1.target_network.set_weights(agent1.main_network.get_weights())
            
            '''# save model
            if step % MODEL_SAVE_FREQ == 0 and step > REPLAY_MEMORY_START_SIZE:
                agent1.main_network.save(save_path + 'agent'+str(agent1.index)+'/wlocal_step'+str(step)+'.h5')
            '''
            agent1.steps_done += 1
                
            #print("avg reward of agent 1 : ",avg_reward1)
                
                
            
    
            #loss_avg2 = keras.metrics.Mean()
            tot_rewards2.append(rewards_tot2)
            avg_reward2 = (avg_reward2*(env.curr_step-1) + rewards_info2[0])/env.total_step
            avg_reward_his2.append(avg_reward2)
            
            # Add experience
            my_replay_memory2.add_experience(action=action2,
                                            frame=next_obs2,
                                            reward=rewards_tot2,
                                            terminal=done2)
            R_comf2_his.append(R_comf2)
            R_eff2_his.append(R_eff2)
            R_safe2_his.append(R_safe2)
            avg_R_comf2 = (avg_R_comf2*(env.curr_step-1) + rewards_info2[1])/env.total_step
            avg_R_comf2_his.append(avg_R_comf2)
            avg_R_eff2 = (avg_R_eff2*(env.curr_step-1) + rewards_info2[2])/env.total_step
            avg_R_eff2_his.append(avg_R_eff2)
            avg_R_safe2 = (avg_R_safe2*(env.curr_step-1) + rewards_info2[3])/env.total_step
            avg_R_safe2_his.append(avg_R_safe2)
            
            if agent2.steps_done > REPLAY_MEMORY_START_SIZE:
                loss_value2 = agent2.train_step_(my_replay_memory2)
                loss_avg2.update_state(loss_value2)
                agent2.update_network()
                
            else:
                loss_avg2.update_state(-1)
                
            loss_avg2_his.append(loss_avg2.result().numpy())
            # Copy network from main to target every NETW_UPDATE_FREQ
            if agent2.steps_done % UPDATE_FREQ == 0 and agent2.steps_done > REPLAY_MEMORY_START_SIZE:
                agent2.target_network.set_weights(agent2.main_network.get_weights())
            '''
            # save model
            if step % MODEL_SAVE_FREQ == 0 and step > REPLAY_MEMORY_START_SIZE:
                agent2.main_network.save(save_path + 'agent'+str(agent2.index)+'/wlocal_step'+str(step)+'.h5')
            '''
            agent2.steps_done += 1
                
            
            
            #loss_avg3 = keras.metrics.Mean()
            tot_rewards3.append(rewards_tot3)
            avg_reward3 = (avg_reward3*(env.curr_step-1) + rewards_info3[0])/env.total_step
            avg_reward_his3.append(avg_reward3)
            
            # Add experience
            my_replay_memory3.add_experience(action=action3,
                                            frame=next_obs3,
                                            reward=rewards_tot3,
                                            terminal=done3)
            R_comf3_his.append(R_comf3)
            R_eff3_his.append(R_eff3)
            R_safe3_his.append(R_safe3)
            avg_R_comf3 = (avg_R_comf3*(env.curr_step-1) + rewards_info3[1])/env.total_step
            avg_R_comf3_his.append(avg_R_comf3)
            avg_R_eff3 = (avg_R_eff3*(env.curr_step-1) + rewards_info3[2])/env.total_step
            avg_R_eff3_his.append(avg_R_eff3)
            avg_R_safe3 = (avg_R_safe3*(env.curr_step-1) + rewards_info3[3])/env.total_step
            avg_R_safe3_his.append(avg_R_safe3)
            
            if agent3.steps_done > REPLAY_MEMORY_START_SIZE:
                loss_value3 = agent3.train_step_(my_replay_memory3)
                loss_avg3.update_state(loss_value3)
                agent3.update_network()
                
            else:
                loss_avg3.update_state(-1)
                
            loss_avg3_his.append(loss_avg3.result().numpy())
            # Copy network from main to target every NETW_UPDATE_FREQ
            if agent3.steps_done % UPDATE_FREQ == 0 and agent3.steps_done > REPLAY_MEMORY_START_SIZE:
                agent3.target_network.set_weights(agent3.main_network.get_weights())
            '''
            # save model
            if step % MODEL_SAVE_FREQ == 0 and step > REPLAY_MEMORY_START_SIZE:
                agent3.main_network.save(save_path + 'agent'+str(agent3.index)+'/wlocal_step'+str(step)+'.h5')
            '''
            agent3.steps_done += 1
            
            
            
            #loss_avg4 = keras.metrics.Mean()
            tot_rewards4.append(rewards_tot4)
            avg_reward4 = (avg_reward4*(env.curr_step-1) + rewards_info4[0])/env.total_step
            avg_reward_his4.append(avg_reward4)
            
            # Add experience
            my_replay_memory4.add_experience(action=action4,
                                            frame=next_obs4,
                                            reward=rewards_tot4,
                                            terminal=done4)
            R_comf4_his.append(R_comf4)
            R_eff4_his.append(R_eff4)
            R_safe4_his.append(R_safe4)
            avg_R_comf4 = (avg_R_comf4*(env.curr_step-1) + rewards_info4[1])/env.total_step
            avg_R_comf4_his.append(avg_R_comf4)
            avg_R_eff4 = (avg_R_eff4*(env.curr_step-1) + rewards_info4[2])/env.total_step
            avg_R_eff4_his.append(avg_R_eff4)
            avg_R_safe4 = (avg_R_safe4*(env.curr_step-1) + rewards_info4[3])/env.total_step
            avg_R_safe4_his.append(avg_R_safe4)
            
            if agent4.steps_done > REPLAY_MEMORY_START_SIZE:
                loss_value4 = agent4.train_step_(my_replay_memory4)
                loss_avg4.update_state(loss_value4)
                agent4.update_network()
                
            else:
                loss_avg4.update_state(-1)
    
            loss_avg4_his.append(loss_avg4.result().numpy())
            # Copy network from main to target every NETW_UPDATE_FREQ
            if agent4.steps_done % UPDATE_FREQ == 0 and agent4.steps_done > REPLAY_MEMORY_START_SIZE:
                agent4.target_network.set_weights(agent4.main_network.get_weights())
            '''
            # save model
            if step % MODEL_SAVE_FREQ == 0 and step > REPLAY_MEMORY_START_SIZE:
                agent4.main_network.save(save_path + 'agent'+str(agent4.index)+'/wlocal_step'+str(step)+'.h5')
            '''
            agent4.steps_done += 1
            
            sumr1 += rewards_tot1
            sumr2 += rewards_tot2
            sumr3 += rewards_tot3
            sumr4 += rewards_tot4
            
            #print(done1,done2,done3,done4)
            if done1+done2+done3+done4:
                #print(str(true_step)+" steps, collied")
                break
            
            else:
                if true_step % 100 == 0 :
                    comm_round += 1
                    # global update
                    #w1,w2,w3,w4 = [1,1,1,1]

                    r_local1 = rewards_tot1
                    c_local1 =cosine_distance(agent1.main_network.get_weights(), g_weights)
                    r_local2 = rewards_tot2
                    c_local2 =cosine_distance(agent2.main_network.get_weights(), g_weights)
                    r_local3 = rewards_tot3
                    c_local3 =cosine_distance(agent3.main_network.get_weights(), g_weights)
                    r_local4 = rewards_tot4
                    c_local4 =cosine_distance(agent4.main_network.get_weights(), g_weights)
                    
                    r_tot = rewards_tot1 + rewards_tot2 + rewards_tot3 + rewards_tot4
                    c_tot = c_local1 + c_local2 + c_local3 + c_local4
                    
                    if c_local1>0.999 and c_local2>0.999 and c_local3>0.999 and c_local4>0.999:
                        w1 = w_r*r_local1/r_tot
                        w2 = w_r*r_local2/r_tot
                        w3 = w_r*r_local3/r_tot
                        w4 = w_r*r_local4/r_tot
                        
                    else:
                        w1 = w_r*r_local1/r_tot + w_c*c_local1/c_tot
                        w2 = w_r*r_local2/r_tot + w_c*c_local2/c_tot
                        w3 = w_r*r_local3/r_tot + w_c*c_local3/c_tot
                        w4 = w_r*r_local4/r_tot + w_c*c_local4/c_tot
                        
                    w_sum = w1+w2+w3+w4
                    w1 = w1/w_sum
                    w2 = w2/w_sum
                    w3 = w3/w_sum
                    w4 = w4/w_sum
            
                    g_weights = w1*np.array(agent1.main_network.get_weights())+w2*np.array(agent2.main_network.get_weights())+w3*np.array(agent3.main_network.get_weights())+w4*np.array(agent4.main_network.get_weights())
                    global_model.target_network.set_weights(g_weights)
                    global_model.main_network.set_weights(g_weights)
                    #global_model.main_network.save(save_path + 'global_model.h5')
                    
                    r_local1_his.append(r_local1)
                    r_local2_his.append(r_local2)
                    r_local3_his.append(r_local3)
                    r_local4_his.append(r_local4)
                    
                    c_local1_his.append(c_local1)
                    c_local2_his.append(c_local2)
                    c_local3_his.append(c_local3)
                    c_local4_his.append(c_local4)
                    
                    w_local1_his.append(w1)
                    w_local2_his.append(w2)
                    w_local3_his.append(w3)
                    w_local4_his.append(w4)
                    
                    # update local model
                    agent1.target_network.set_weights(g_weights)
                    agent2.target_network.set_weights(g_weights)
                    agent3.target_network.set_weights(g_weights)
                    agent4.target_network.set_weights(g_weights)
                    
                    # energy consumption
                    e_m, e_comp , e_comm,e = env.total_energy(STEP_PER_EPOCH=steps_per_epoch, G_UPDATE_FREQ=GLOBAL_UPDATE_FREQ, N_AGENT=NB_AGENT)
                    total_energy_drone += e
                    tot_e_comm+=e_comm
                    tot_e_comp+=e_comp
                    tot_e_m +=e_m
                    
                    energy_his.append(e)
                    #avg_energy = (avg_energy*(env.curr_step-1) + e)/env.total_step
                    #avg_energy_his.append(avg_energy)
            
                    
                    # delay
                    d1,d2,d3,d4 = env.total_delay(VEH_NUM=env.numVehicles,N_AGENT=NB_AGENT)
                    total_delay_agent1 += d1
                    total_delay_agent2 += d2
                    total_delay_agent3 += d3
                    total_delay_agent4 += d4
                    
                    delay_his1.append(d1)
                    delay_his2.append(d2)
                    delay_his3.append(d3)
                    delay_his4.append(d4)
                    
                    delay_his1.append(d1)
                    delay_his2.append(d2)
                    delay_his3.append(d3)
                    delay_his4.append(d4)
                
                else:
                    continue
        
        # record reward of each episode
        #episode_return = 0
        
        #env.close()
        #traci.close()
        #print("close and reset env")
        env.close()
        
        next_obs1,next_obs2,next_obs3,next_obs4 = env.reset(gui=False, numVehicles=150) 
    
        state1 = next_obs1[0].copy()
        state2 = next_obs2[0].copy()
        state3 = next_obs3[0].copy()
        state4 = next_obs4[0].copy()
        done1 = done2 = done3 = done4 = False
        
        step_his.append(true_step)
        
        sumr1_his.append(sumr1)
        sumr2_his.append(sumr2)
        sumr3_his.append(sumr3)
        sumr4_his.append(sumr4)
        # save model
        '''if step % MODEL_SAVE_FREQ == 0 and step > REPLAY_MEMORY_START_SIZE:
            global_model.main_network.save(save_path + 'global_model'+str(step)+'.h5')'''
except KeyboardInterrupt:
    env.close()
    fig, ax = plt.subplots()
    ax.plot(sumr1_his,label='agent1')
    ax.plot(sumr1_his,label='agent2')
    ax.plot(sumr1_his,label='agent3')
    ax.plot(sumr1_his,label='agent4')
    ax.set_xlabel('epoch')
    ax.set_ylabel('reward')
    ax.set_title('Reward of agents')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(loss_avg1_his,label='agent1')
    ax.plot(loss_avg2_his,label='agent2')
    ax.plot(loss_avg3_his,label='agent3')
    ax.plot(loss_avg4_his,label='agent4')
    ax.set_xlabel('epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss of agents')
    ax.legend()
    plt.show()


global_model.main_network.save(save_path + 'new_4clients_weighted_global_model_freq_'+str(GLOBAL_UPDATE_FREQ)+'_'+str(steps_per_epoch*epochs)+'e.h5')

fig, ax = plt.subplots()
ax.plot(loss_avg1_his,label='agent1')
ax.plot(loss_avg2_his,label='agent2')
ax.plot(loss_avg3_his,label='agent3')
ax.plot(loss_avg4_his,label='agent4')
ax.set_xlabel('epoch')
ax.set_ylabel('Loss')
ax.set_title('Loss of agents')
ax.legend()
plt.show()
