import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras.layers as kl
import numpy as np
import logging
import datetime
import os
import random
import math
from replayMemory import ReplayMemory
from tqdm import tqdm

def LaneChangeModel(num_actions, input_len, lr):

    model = keras.Sequential()
    model.add(kl.Dense(64, input_shape=(input_len,), activation='relu'))
    model.add(kl.Dense(128, activation='relu'))
    model.add(kl.Dense(64, activation='relu'))
    model.add(kl.Dense(num_actions,activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='mse')

    return model


class PGAgent:
    def __init__(self, fn=None, lr=0.01, gamma=0.95, batch_size=32):
        # Coefficients are used for the loss terms.


        self.state_shape=48 # the state space
        self.action_shape=3 # the action space
        self.gamma=gamma # decay rate of past observations
        self.alpha=1e-4 # learning rate in the policy gradient
        self.learning_rate=0.01 # learning rate in deep learning
        self.network = LaneChangeModel(num_actions=3, input_len=self.state_shape, lr=lr)

        # record observations
        self.states=[]
        self.gradients=[]
        self.rewards=[]
        self.probs=[]
        self.discounted_rewards=[]
        self.total_rewards=[]


    def hot_encode_action(self, action):
        
        action_encoded = np.zeros(self.action_shape, np.float32)
        action_encoded[action] = 1
        return action_encoded

    def remember(self, state, action, action_prob, reward):
        '''stores observations'''
        encoded_action=self.hot_encode_action(action)
        self.gradients.append(encoded_action-action_prob)
        self.states.append(state)
        self.rewards.append(reward)
        self.probs.append(action_prob)

    def act(self, state, network):
        # we need to do exploration vs exploitation
        # epsilon-greedy exploration
        '''samples the next action based on the policy probabilty distribution
        of the actions'''

        # transform state
        state=state.reshape([1, state.shape[0]])
        # get action probably
        action_probability_distribution=network.predict(state).flatten()
        #print(action_probability_distribution)
        # norm action probability distribution
        action_probability_distribution/=np.sum(action_probability_distribution)

        # sample action
        action=np.random.choice(self.action_shape,1,p=action_probability_distribution)[0]
        #print(action)
        return action, action_probability_distribution

    def get_discounted_rewards(self, rewards):
        '''Use gamma to calculate the total reward discounting for rewards
        Following - \gamma ^ t * Gt'''

        discounted_rewards=[]
        cumulative_total_return=0
        # iterate the rewards backwards and and calc the total return
        for reward in rewards[::-1]:
          cumulative_total_return=(cumulative_total_return*self.gamma)+reward
          discounted_rewards.insert(0, cumulative_total_return)

        # normalize discounted rewards
        mean_rewards=np.mean(discounted_rewards)
        std_rewards=np.std(discounted_rewards)
        norm_discounted_rewards=(discounted_rewards-
                              mean_rewards)/(std_rewards+1e-7) # avoiding zero div

        return norm_discounted_rewards


    def update_policy(self):
        '''Updates the policy network using the NN model.
        This function is used after the MC sampling is done - following
        \delta \theta = \alpha * gradient + log pi'''

        # get X
        states=np.vstack(self.states)

        # get Y
        gradients=np.vstack(self.gradients)
        rewards=np.vstack(self.rewards)
        discounted_rewards=self.get_discounted_rewards(rewards)
        gradients*=discounted_rewards
        gradients=self.alpha*np.vstack([gradients])+self.probs

        history=self.network.train_on_batch(states, gradients)

        self.states, self.probs, self.gradients, self.rewards=[], [], [], []

        return history

    def train(self, env, episodes=100, rollout_n = 64):
        '''train the model
        episodes - number of training iterations
        rollout_n- number of episodes between policy update'''

        total_rewards = np.zeros(episodes)
        for episode in tqdm(range(episodes)):
            #print(episode)
            next_obs = env.reset(gui=True, numVehicles=150)
            state = next_obs.copy()
            done = False
            episode_reward = 0

            while not done:

                action, prob = self.act(state,self.network)
                next_obs, rewards_info, done, collision = env.step(action)
                reward_tot, R_comf, R_eff, R_safe = rewards_info
                #print(done)
                self.remember(state, action, prob, reward_tot)

                state = next_obs
                episode_reward += reward_tot

                if done:
                    #update policy
                    if episode%rollout_n == 0:
                        history = self.update_policy()

            total_rewards[episode] = episode_reward
            
            env.close()
        self.total_rewards = total_rewards
        
        save_path = "PG_models/"
        self.network.save(save_path + 'icc_pg_100its_noglobal.h5')
        print("Number of real lane change:", env.nb_lc)
        print("Number of collisions:", env.total_collision)
        print("Current step:", env.curr_step)

        
        #env.close()

        return 0
