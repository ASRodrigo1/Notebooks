import warnings
warnings.simplefilter('ignore')
import tensorflow as tf
print("Is GPU available?", tf.test.is_gpu_available())
print("TF version:", tf.__version__)
print("Keras version:", tf.keras.__version__)

import gym
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam

class NN(Model):
    def __init__(self, n_actions, input_shape):
        super(NN, self).__init__()
            
        self.conv1 = Conv2D(filters=64, kernel_size=4, strides=1, activation='relu', input_shape = (*input_shape, ), data_format='channels_first')
        self.conv2 = Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', data_format='channels_first')
        self.conv3 = Conv2D(filters=64, kernel_size=2, strides=2, activation='relu', data_format='channels_first')
        self.flatten = Flatten(data_format='channels_first')
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(128, activation='relu')
        self.V = Dense(1, activation=None)
        self.A = Dense(n_actions, activation=None)
    
    def call(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)
        
        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))
        
        return Q
    
    def advantage(self, state):
        x = self.conv1(state)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        A = self.A(x)
        
        return A

class Agent():
    def __init__(self, input_shape, n_actions, mem_size, eps, eps_min, eps_dec, gamma, q_eval_name, q_next_name, 
                 replace_freq, lr=0.0005):
        self.Q_eval = NN(n_actions, input_shape)
        self.Q_next = NN(n_actions, input_shape)
        self.Q_eval.compile(optimizer=Adam(lr=lr), loss='mse')
        self.Q_next.compile(optimizer=Adam(lr=lr), loss='mse')
        self.memory = deque(maxlen=mem_size)
        self.eps = eps
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.gamma = gamma
        self.replace = replace_freq
        self.action_space = [i for i in range(n_actions)]
        self.steps = 0
        self.input_shape = input_shape
        self.q_eval_name = q_eval_name
        self.q_next_name = q_next_name
    
    def store(self, state, action, reward, n_state, done):
        pack = [np.expand_dims(state, axis=0), action, reward, np.expand_dims(n_state, axis=0), done]
        self.memory.append(pack)
    
    def take_data(self, batch_size):
        pack = random.sample(self.memory, batch_size)
        states = []
        actions = []
        rewards = []
        n_states = []
        dones = []
        for i in range(batch_size):
            states.append(pack[i][0])
            actions.append(pack[i][1])
            rewards.append(pack[i][2])
            n_states.append(pack[i][3])
            dones.append(pack[i][4])
        return states, actions, rewards, n_states, dones
    
    def choose_action(self, state):
        if np.random.random() > self.eps:
            return np.argmax(self.Q_eval.advantage(state))
        return np.random.choice(self.action_space)
    
    def decay_eps(self): 
        self.eps = self.eps - self.eps_dec if self.eps > self.eps_min else self.eps_min
    
    def replace_weights(self):
        if not (self.steps % self.replace):
            self.Q_next.set_weights(self.Q_eval.get_weights())
    
    def upgrade(self, batch_size=64):
        if len(self.memory) >= 4*batch_size:
            states, actions, rewards, n_states, dones = self.take_data(batch_size)
            
            self.replace_weights()
            
            act = [np.argmax(self.Q_eval(n_states[i])) for i in range(batch_size)]
            q_next = [self.Q_next(n_states).numpy()[0][act[i]] for i in range(batch_size)]
            q_target = [self.Q_eval(states[i]).numpy()[0] for i in range(batch_size)]
            
            for i in range(batch_size):
                q_target[i][actions[i]] = rewards[i] + self.gamma*q_next[i]*(1 - dones[i])
            
            states = np.reshape(states, (batch_size, *self.input_shape))
            
            self.Q_eval.train_on_batch(np.array(states), np.array(q_target))
            
            self.decay_eps()
            self.steps += 1

    def save(self):
        self.Q_eval.save_weights(self.q_eval_name)

def transform(state):
    state = cv2.resize(state, (84, 84))
    state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY)
    state = np.reshape(state, (1, 1, 84, 84))
    state = np.array(state, np.float32)
    return state

if __name__ == '__main__':
	env = gym.make('Breakout-v0')
	n_games = 600
	agent = Agent(input_shape=(1, 84, 84), n_actions=env.action_space.n, mem_size=10000, eps=1.0, 
              	eps_min=0.001, eps_dec=0.001, gamma=0.99, q_eval_name='Q_eval.h5', q_next_name='Q_next.h5', 
              	replace_freq=500)
	best_score = -500
	scores = []
	means = []
	eps = []
	for i in range(n_games):
	    done = False
	    state = env.reset()
	    state = transform(state)
	    score = 0
	    while not done:
	        action = agent.choose_action(np.expand_dims(state, axis=0))
	        for i in range(4):
	            n_state, reward, done, _ = env.step(action)
	            n_state = transform(n_state)
	            agent.store(state, action, reward, n_state, int(done))
	            state = n_state
	        score += reward
	        agent.upgrade()
	    scores.append(score)
	    mean = np.mean(scores[-20:])
	    means.append(mean)
	    eps.append(agent.eps)
	    if mean > best_score:
	        agent.save()
	        best_score = mean
	    print('episode: ', i+1, '   score: ', score, '   eps:  %.3f' %agent.eps)
	env.close()