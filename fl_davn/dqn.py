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
import matplotlib.pyplot as plt
from tqdm import tqdm

def LaneChangeModel(num_actions, input_len):
    input = kl.Input(shape=(input_len))
    hidden1 = kl.Dense(64, activation='relu')(input)
    hidden2 = kl.Dense(128, activation='relu')(hidden1)
    hidden3 = kl.Dense(64, activation='relu')(hidden2)
    state_value = kl.Dense(1)(hidden3)
    state_value = kl.Lambda(lambda s: keras.backend.expand_dims(s[:, 0], -1), output_shape=(num_actions,))(state_value)

    action_advantage = kl.Dense(num_actions)(hidden3)
    action_advantage = kl.Lambda(lambda a: a[:, :] - keras.backend.mean(a[:, :], keepdims=True), output_shape=(num_actions,))(
        action_advantage)

    X = kl.Add()([state_value, action_advantage])

    model = keras.models.Model(input, X, name='LanChangeModel')
    return model


class DQNAgent:
    def __init__(self, fn=None, lr=0.001, gamma=0.95, batch_size=32,ind=0):
        # Coefficients are used for the loss terms.
        self.index = ind
        self.gamma = gamma
        self.lr = lr
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.checkpoint_dir = 'checkpoints/'
        self.model_name = 'DQN'+str(self.index)
        self.model_dir = self.checkpoint_dir + self.model_name
        self.log_dir = 'logs/'
        self.train_log_dir = self.log_dir + self.model_name
        self.create_log_dir()
        self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        self.fn = fn
        self.eps_threshold=0.2
        self.steps_done = 0
        self.EPS_DECAY = 100
        self.batch_size = batch_size
        self.TAU = 0.08
        # Parameter updates
        self.loss = keras.losses.Huber()
        self.optimizer = tf.optimizers.Adam(learning_rate=self.lr)
        self.main_network = LaneChangeModel(num_actions=3, input_len=48)
        self.target_network = LaneChangeModel(num_actions=3, input_len=48)

    def create_log_dir(self):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        if not os.path.exists(self.train_log_dir):
            os.mkdir(self.train_log_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

    def act(self, state, network):
        # we need to do exploration vs exploitation
        # epsilon-greedy exploration
        if np.random.rand() < self.eps_threshold:
            action = random.randint(0, 2)
        else:
            action = network.predict(np.expand_dims(state, axis=0))
            action = np.argmax(action)
        return action

    def train_step_(self, replay_memory):
        states, actions, rewards, new_states, terminal_flags = replay_memory.get_minibatch()
        q_vals = self.main_network(new_states)
        actions = np.argmax(q_vals, axis=1)
        # The target network estimates the Q-values (in the next state s', new_states is passed!)
        # for every transition in the minibatch
        q_vals = self.target_network(new_states)
        # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
        # if the game is over, targetQ=rewards
        q_vals = np.array([q_vals[num, action] for num, action in enumerate(actions)])
        target_q = rewards + (self.gamma * q_vals * (1 - terminal_flags))
        loss = self.main_network.train_on_batch(states, target_q)
        return loss

    def update_network(self):
        # update target network parameters slowly from primary network
        for t, e in zip(self.main_network.trainable_variables, self.target_network.trainable_variables):
            t.assign(t * (1 - self.TAU) + e * self.TAU)

    def train(self, env, steps_per_epoch=64, epochs=100):
        # Every four actions a gradient descend step is performed
        UPDATE_FREQ = 4
        # Number of chosen actions between updating the target network.
        NETW_UPDATE_FREQ = 10000
        # Replay mem
        REPLAY_MEMORY_START_SIZE = 33
        # Create network model
        self.main_network.compile(optimizer=tf.keras.optimizers.Adam(), loss='mse')
        # Replay memory
        my_replay_memory = ReplayMemory()
        avg_reward = 0
        avg_speed = 0
        tot_rewards = []
        avg_reward_his = []
        avg_speed_his = []
        # Metrics
        loss_avg = keras.metrics.Mean()

        # Training loop: collect samples, send to optimizer, repeat updates times.
        
        first_epoch = 0
        try:
            for epoch in tqdm(range(first_epoch, epochs)):
                #print(epoch)
                env.close()
                next_obs = env.reset(gui=True, numVehicles=150)
                for step in range(steps_per_epoch):
                    # curr state
                    state = next_obs.copy()
                    # get action
                    action = self.act(state, self.main_network)
                    # do step
                    next_obs, rewards_info, done, collision = env.step(action)
                    '''if done:
                        break
                    else:'''
                        
                    # process obs and get rewards
                    rewards_tot, R_comf, R_eff, R_safe = rewards_info
                    tot_rewards.append(rewards_tot)
                    avg_reward = (avg_reward*(env.curr_step-1) + rewards_info[0])/env.total_step
                    avg_reward_his.append(avg_reward)
                    avg_speed = (avg_speed*(env.curr_step-1) + state[3])/env.total_step
                    avg_speed_his.append(avg_speed)
                    # Add experience
                    my_replay_memory.add_experience(action=action,
                                                    frame=next_obs,
                                                    reward=rewards_tot,
                                                    terminal=done)

                    # Train every UPDATE_FREQ times
                    if self.steps_done > REPLAY_MEMORY_START_SIZE:
                        loss_value = self.train_step_(my_replay_memory)
                        loss_avg.update_state(loss_value)
                        self.update_network()
                    else:
                        loss_avg.update_state(-1)
                    # Copy network from main to target every NETW_UPDATE_FREQ
                    if step % NETW_UPDATE_FREQ == 0 and step > REPLAY_MEMORY_START_SIZE:
                        self.target_network.set_weights(self.main_network.get_weights())

                    self.steps_done += 1


            #save_path = "DQN_models/"
            #self.main_network.save(save_path + 'DQNmodel_.h5')
            print("Number of real lane change:", env.nb_lc)
            print("Number of collisions:", env.total_collision)
            print("Current step:", env.curr_step)
            print("Average speed:", avg_speed)
            print("avg reward:",avg_reward)
            
        except KeyboardInterrupt:
            save_path = "DQN_models/"
            self.main_network.save(save_path + 'DQNmodel000.h5')

            '''print("Number of real lane change:", env.nb_lc)
            print("Number of collisions:", env.total_collision)
            print("Current step:", env.curr_step)
            print("Average speed:", avg_speed)'''
            print("avg reward:",avg_reward)


        #print("Average speed:", avg_speed)
        #print("avg reward:",avg_reward)

        #env.close()
        return self.target_network.get_weights()