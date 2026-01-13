import sys
import logging
import os
import multiprocessing as mp


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import copy
import random
from collections import deque
import numpy as np
import gen_data_ref as gen

from Env_ref import Env
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras import regularizers
from keras import backend as K

from PER_ref import Memory



class Agent:

    def __init__(
        self,
        n_blocks,
        n_trains,
        n_node,
        n_alterArcSet,
        load_file=None,
        learning_rate    = 0.001,   
        discount_factor  = 0.9,     
        epsilon_decay    = 0.9999,     
        epsilon_min      = 0.01,      
        batch_size       = 256,
        train_start      = 10000,
        train_freq       = 16,
        update_freq      = 64,
        l2_reg           = 0.001,
        memory_size      = 100000,
        decay_freq       = 16,
        epsilon_start    = 1.0,
    ):

        self.n_blocks = n_blocks
        self.n_trains = n_trains
        self.n_node = n_node
        self.n_alterArcSet = n_alterArcSet

        self.state_size = 2 + self.n_blocks * self.n_blocks + 9 * self.n_blocks
        self.action_size = self.n_blocks + 1

        self.learning_rate = learning_rate
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.decay_freq = decay_freq
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.train_start = train_start
        self.train_freq = train_freq
        self.update_freq = update_freq
        self.memory_size = memory_size
        self.l2_reg = l2_reg

        self.memory = Memory(self.memory_size)

        self.model = self.build_model()
        if load_file is not None:
            print("Load weights: ")
            self.model.load_weights(load_file)

        self.target_model = self.build_model()
        self.update_target_model()

        self.states, self.actions, self.rewards = [], [], []



    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        model = Sequential()
        model.add(Dense(
            1024,
            input_dim=self.state_size,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            bias_regularizer=regularizers.l2(self.l2_reg),
            kernel_initializer="random_uniform",
            bias_initializer="zeros",
        ))
        model.add(Dense(
            512,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            bias_regularizer=regularizers.l2(self.l2_reg),
            kernel_initializer='random_uniform',
            bias_initializer='zeros',
        ))
        model.add(Dense(
            256,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            bias_regularizer=regularizers.l2(self.l2_reg),
            kernel_initializer='random_uniform',
            bias_initializer='zeros',
        ))
        model.add(Dense(
            128,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            bias_regularizer=regularizers.l2(self.l2_reg),
            kernel_initializer='random_uniform',
            bias_initializer='zeros',
        ))
        model.add(Dense(
            self.action_size,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            bias_regularizer=regularizers.l2(self.l2_reg),
            kernel_initializer='random_uniform',
            bias_initializer='zeros',
        ))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model


    def append_sample(self, state, action, reward, next_state, is_terminal):
        state_ = np.array(state, copy=True)

        normalized_state = np.reshape(self.state_normalization(state_),[1, self.state_size])

        old_val = self.model.predict(normalized_state, verbose=0)[0][action]

        if is_terminal:
           pass
        else:
           next_state_ = np.array(next_state, copy=True)
           normalized_next_state = np.reshape(self.state_normalization(next_state_),[1,self.state_size])
           target_val = self.target_model.predict(normalized_next_state, verbose=0)[0]

        if not is_terminal:
           new_val = reward + self.discount_factor*np.max(target_val)
        else:
           new_val = reward

        error = np.abs(old_val - new_val)
        self.memory.add(error, (state, action, reward, next_state))

    def get_action(self, state_, current_postion_, next_postion_):
        state = copy.deepcopy (state_)
        current_postion = copy.deepcopy (current_postion_)
        next_postion = copy.deepcopy(next_postion_)
        remove_set = {0}
        postion_without_zero = list(set(current_postion) - set(remove_set))
        if np.random.rand () <= self.epsilon:
            self.action_type="exploration"
            avail_positions = []
            for i in range(len(current_postion)):
                tc = current_postion[i]
                if tc != 0 and next_postion[i] not in postion_without_zero:
                    tc_index = env.tc_list.index(tc)
                    avail_positions.append(tc_index)
            if len(avail_positions) == 0: return self.n_blocks 
            randomIndex = random.randrange(0, len(avail_positions))
            return avail_positions[randomIndex]

        else:
            self.action_type="exploitation"
            normalized_state = self.state_normalization (state)
            normalized_state = np.reshape (normalized_state, [1, self.state_size])
            q_values = self.model.predict (normalized_state, verbose=0)[0]
            avail_values = []
            avail_positions = []
            for i in range(len(current_postion)):
                tc = current_postion[i]
                if tc != 0 and next_postion[i] not in postion_without_zero:
                    tc_index = env.tc_list.index(tc)
                    avail_values.append(q_values[tc_index])
                    avail_positions.append(tc_index)

            if len(avail_positions) == 0: return self.n_blocks 
            max_position = np.argmax (avail_values)
            return avail_positions[max_position]

    def state_normalization(self, state_):
        state = copy.deepcopy(state_)
        #max_state = np.max(state)
        #state = state / max_state
        return state.reshape((1, self.state_size))

    def train_model(self, global_step, done):
        if self.epsilon >= self.epsilon_min and global_step % self.decay_freq  == 0:
            self.epsilon *= self.epsilon_decay
        # tiny periodic exploration bump (every ~4000 global steps)
        if global_step % 4000 == 0:
            self.epsilon = max(self.epsilon, 0.15) 


        mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
      

        # print("mini_batch", mini_batch)
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards = [], []

        for i in range(self.batch_size):
            #print("mini_batch[i][0]", mini_batch[i][0])
            states[i] = self.state_normalization(mini_batch[i][0])
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = self.state_normalization(mini_batch[i][3])


        target = self.model.predict(states, verbose=0)
        target_value = self.target_model.predict(next_states, verbose=0)
        errors = []
        # print(target.shape)
        # print(target_value.shape)

        for i in range(self.batch_size):
            absolute_td_error = np.abs(target[i][actions[i]]-rewards[i]+self.discount_factor*(np.amax(target_value[i])))
            if done:
               errors.append(np.abs(target[i][actions[i]]-rewards[i]))
               target[i][actions[i]] = rewards[i]
            else:
               errors.append(np.abs(target[i][actions[i]]-rewards[i]-self.discount_factor*np.amax(target_value[i])))
               target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_value[i]))

        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx,errors[i])

        hist = self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)
        #print("===== MSE loss:",hist.history['loss'])
        #f.write(str(global_step) + "\t" + str(hist.history['loss']) + "\n")



if __name__ == "__main__":

    print("Number of processors: ", mp.cpu_count())
    load_file = None

    #if len (sys.argv) == 6:
    #    load_file = sys.argv[5]

    #network_file = str(sys.argv[1])
    #train_file = str(sys.argv[2])

    #network_file = "simple_network.txt"
    #train_file = "simple_train.txt"
    network_file = "Python/simple_network.txt"
    train_file = "Python/simple_train_case5.txt"
    max_delay = 100
    conflict_delay_cap = 120   
    early_cap = 120


    env = Env(network_file, train_file, max_delay,  conflict_delay_cap, early_cap, training=False)

    n_blocks = len(env.tc_list)
    n_max_trains = len(env.train_ids)
    n_node = len(env.nodes)
    n_alterArcSet = len(env.alterArcSet)

    #f = open("20220218_MSE_result_cap_final-"+str(n_blocks)+"-"+str(n_max_trains)+".txt", 'w')

    agent = Agent (n_blocks, n_max_trains, n_node, n_alterArcSet, load_file)

    scores, episodes, objs = [], [], []
    global_step = 0

    EPISODES = 1
    ITERATION_PER_EPISODE = n_max_trains*n_blocks*n_blocks

    e = 0
    while True:
        f = open("20220402_MSE_result_softearlycap_epsilon_strict-"+str(n_blocks)+"-"+str(n_max_trains)+".txt", 'a')
        episode_return = 0.0

        e += 1
        env.episode_idx = e

        state = env.reset ()
        initial_obj = env.current_lp_cost
        done = False
        current_best = initial_obj
        prev_action = -1
        for iteration in range(ITERATION_PER_EPISODE):
            global_step += 1
            action = agent.get_action (state, env.current_postion, env.next_pos)

            next_state, reward, is_terminal = env.step (action)
            episode_return += reward

            if is_terminal != 'Pass':
                agent.append_sample(state, action, reward, next_state, is_terminal)
            pre_action = action

            if is_terminal:
                done = True

            if (global_step % agent.train_freq == 0 and global_step >= agent.train_start):
                agent.train_model(global_step, done)
                if global_step % agent.update_freq == 0:
                    agent.update_target_model()

            if env.current_lp_cost < current_best: current_best = env.current_lp_cost
            state = copy.deepcopy (next_state)

            if done: break


        objs.append (env.current_lp_cost)
        print ("the end of episode:", e, "\tglobal step:", global_step, "\tcurrent obj.:", env.current_lp_cost,"\tepsilon:", agent.epsilon)
        f.write("episode\t" + str(e) + "\t" + str(-1*env.current_lp_cost) + "\t" + str(env.last_max_train_end_delay) + "\t" + str(getattr(env, 'last_max_train_end_early', 0.0)) + "\t" + str(env.last_episode_return) + "\t" +  str(getattr(env, 'last_max_conflict_increment', 0.0)) + "\t" +
            str(getattr(env, 'last_max_early_increment', 0.0)) + "\n")
        
        #agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.epsilon_min)

        if e % 100 == 0:
            agent.model.save_weights (
                 "./save_model_softearlycap_epsilon_strict/20220402_DQN_" + str (n_blocks) + "_" + str (n_max_trains) + "_" + str (
                     e) + ".weights.h5")

        if e % 100 == 0:
            print("Fail %", round(env.n_Fail / 100, 2), "\t Feasible %", round(env.n_Feasible / 100, 2), "\t Optimal %", round(env.n_Optimal / 100, 2))
            env.n_Fail = 0
            env.n_Feasible = 0
            env.n_Optimal = 0

        if agent.epsilon < agent.epsilon_min:   break