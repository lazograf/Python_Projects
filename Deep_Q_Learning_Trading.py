
# Deep Q-Learning

import tensorflow as tf
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

import keras
config = tf.ConfigProto( device_count = {'GPU': 0} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set()

plt.style.use('ggplot')

from keras import regularizers
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.initializers import glorot_normal

import pandas_datareader.data as web
df = web.DataReader('JPM',data_source="yahoo",start="03/01/1980",end="30/6/2019").dropna()  # JP Morgan stock

print(df.head())
df.shape

#rounding dataframe for even more discrete states

df = df.round(2)
df.head()

from collections import deque
import random

class DQN_Agent:
  
    def __init__(self, state_size, window_size, diff_price, batch_size, ): 
      
      
        self.state_size = state_size
        self.window_size = window_size
        self.diff_price = diff_price
        self.action_size = 3
        self.batch_size = batch_size
        self.memory = deque(maxlen = 10000)
        self.portfolio = []
        self.commission_fee = 4.96 #Found on web: Interactive Brokers $1.00 | E-Trade  $6.95 | Fidelity $4.95 | Schwab $4.95 | TD Ameritrade $6.95 } -> } sum = 24.8, average = 24.8/5 = 4.96

        self.gamma = 0.95
        self.epsilon = 0.6 
        self.epsilon_min = 0.01 
        self.epsilon_decay = 0.997 

        
        self.optim = Adam(lr = 1e-3)
        self.model = self.brain(self.optim)
        
    def brain(self, optim):
        model = Sequential()
        
        
        model.add(Dense(64, input_dim= self.state_size,activation='relu', kernel_initializer = 'glorot_normal', kernel_regularizer= regularizers.l2(1e-6)))
         
        #model.add(Dense(32,activation='relu', kernel_regularizer= regularizers.l2(1e-6)))
        model.add(Dense(16,activation='relu', kernel_initializer = 'glorot_normal', kernel_regularizer= regularizers.l2(1e-6)))
        
        
        model.add(Dense(self.action_size,activation='linear'))
        model.compile(loss= tf.losses.huber_loss, metrics = ['mean_squared_error'], optimizer= optim)
        
        model.summary()
        return model
      
       
      
        
        
    def act(self, state): #following an espilon-greedy policy with probability of exploitation [1-epsilon] and exploration [epsilon]
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
                  
        return np.argmax(self.model.predict(state)[0]) # keras model.predict() outputs in the form array([[-0.985]], that's why we use the [0] to access 
                  
        
        # Reminder: In every function only one "return" is returned!
    
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
    
    def get_state(self, t):
        window_size = self.window_size
        d = t - window_size
        
        if d>=0:
          window = self.diff_price[d:t+1]
          #print(window,len(window)) #for debugging
        else:
          padding = abs(d)*[self.diff_price[0]]
          window = padding + self.diff_price[0 : t + 1]
          
          #print("Dbg: else", window,len(window)) #for debugging
        
        state_series = []
        for i in range(window_size): # an integer/length of the window
            #The number of the following differences will be window_size - 1 . E.g. I have a vector of 11 values, the vector of their differences will be 10
            state_series.append(window[i + 1] - window[i]) # calculates a simple first differences for the closing prices in the window: closing_price[t] - closing_price[t-1]
            
            
        #print('\n',np.array([state_series]),len(np.array([state_series]))) #for debugging
        return np.array([state_series])

    
    def sample_data(self, x):
        return random.sample(list(x), min(batch_size,len(self.memory))) # samples self.batch_size number of tuples from memory
             
              
      
    def replay(self, batch_size, random_sampling= False):
        
        mini_batch = []
        
        if random_sampling is False:
            l = len(self.memory)
            for i in range(l - batch_size, l):
                mini_batch.append(self.memory[i])
        else:
            mini_batch = [i for i in self.sample_data(self.memory)]
        
        assert len(mini_batch) == batch_size
        exp_replay_size = len(mini_batch)
        
        buffer_states = np.empty((exp_replay_size, self.state_size))
        buffer_Qvalues = np.empty((exp_replay_size, self.action_size))
        states = np.array([s[0][0] for s in mini_batch])
        new_states = np.array([s[3][0] for s in mini_batch])
        
        Q = self.model.predict(states)
        Q_new = self.model.predict(new_states)
        
        
        
        for i in range(len(mini_batch)):
          
            state, action, reward, next_state, done = mini_batch[i]
            
            target = Q[i]
            target[action] = reward
            if not done: #WILL EXECUTE IF DONE IS FALSE
                target[action] = target[action] + self.gamma * np.amax(Q_new[i]) #Bellman equation
            
            
            buffer_states[i] = state 
            buffer_Qvalues[i] = target
              
     
        history = keras.callbacks.History()
        self.model.fit(buffer_states,buffer_Qvalues, epochs=1, verbose=0, callbacks = [history])
        online_train_loss = history.history['loss'][0]
                      
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay #WE CHANGE THE GENERAL self.epsilon meaning the first episode will be pretty random and for a number of states,
                                               #but in the next episode, epsilon will have been lowered and even the first states will be examined with less randomness by the Agent
        
            #print(self.epsilon) #for debugging
        
        return online_train_loss
    

    

    def train(self, episodes, checkpoint, initial_capital):
        for i in range(episodes):
            total_profit = 0
            portfolio = [] 
            state = self.get_state(0) #Initializing the state variable
            present_capital = initial_capital
            
            for t in range(0,len(self.diff_price)): 
                
                action = self.act(state) # AGENT acts based on np.amax()
                
                if t == len(self.diff_price)-1 :
                    continue
                else:
                    next_state = self.get_state(t + 1)
                
                if action == 1 and present_capital >= self.diff_price[t] and t < (len(self.diff_price) - 10): 
                    portfolio.append(self.diff_price[t])                                                              
                    present_capital -= self.diff_price[t]
                    #total_profit -= self.commission_fee
                    #present_capital += total_profit
                    
                elif action == 2 and len(portfolio) > 0:
                    bought_price = portfolio.pop(0)
                    total_profit += self.diff_price[t] - bought_price
                    #total_profit -= self.commission_fee
                    present_capital += total_profit
                   
                                  
                    
                trading_reward = ((present_capital - initial_capital) / initial_capital) #reward tuple
                if t == len(self.diff_price) - 1:
                    done = True
                elif present_capital < 0 or present_capital == 0:
                    done = True
                else:
                    done = False
                
                self.memory.append((state, action, trading_reward, 
                                    next_state, done))
                
                batch_size = min(self.batch_size, len(self.memory))
                loss = self.replay(batch_size, random_sampling= False ) # States are serially processed if random_sampling == False and that's makes more sense in terms of selling bought stocks
                
                if done == True:
                  if t < len(self.diff_price)-1:
                    print('*** Final state was reached *before* the last price-state (!!!) ***')
                    break
                
                  
                state = next_state
            if (i+1) % checkpoint == 0:
              print('\n','* Final state was reached (!) - End of episode *')
              print('---------------------------------------------------------------------------------------------------')
              print('episode:', i+1, ' -- ', 'online_loss: ', round(loss,10), ' -- ', '\n')                    
                    
              print('sum of rewards: %.3f, total capital: %f, portfolio status: %s'%(total_profit, present_capital, portfolio))
                                                                               
              print('\n', 'memory length is {}'.format(len(self.memory)))
              
              print('---------------------------------------------------------------------------------------------------', '\n')
              



    def positions(self, initial_capital_p):
        initial_capital_p = initial_capital_p
        present_capital_p = initial_capital_p
        states_sell = []
        states_buy = []
        portfolio_p = []
        state = self.get_state(0)
        
        for t in range(0, len(self.diff_price) - 1): 
            
            action = self.act(state)
            next_state = self.get_state(t + 1)
            
            if action == 1 and present_capital_p >= self.diff_price[t]: #and t < (len(self.diff_price) -10):
                portfolio_p.append(self.diff_price[t])
                present_capital_p -= self.diff_price[t]
                #present_capital_p -= self.commission_fee
                states_buy.append(t)
                print('Day %d: Agent buys 1 unit at price %.2f, account balance %.2f'% (t, self.diff_price[t], present_capital_p))
                
                
            elif action == 2 and len(portfolio_p)>0:
                bought_price = portfolio_p.pop(0)
                present_capital_p += self.diff_price[t] - bought_price
                #present_capital_p -= self.commission_fee
                states_sell.append(t)
                
                investment_performance = ((close[t] - bought_price) / bought_price) * 100
                
                print('Day %d, sell 1 unit at price %f, investment performance: %f %%, account balance %f,' % (t, close[t], investment_performance, present_capital_p))
            
            state = next_state
        total_invest_perf = ((present_capital_p - initial_capital_p) / initial_capital_p) * 100
        total_account_gains = present_capital_p - initial_capital_p
        return states_buy, states_sell, total_invest_perf, total_account_gains

close = df.Close.values.tolist()
initial_capital = 10000
window_size = 10
batch_size = 64
RL_Agent = DQN_Agent(state_size = window_size, 
              window_size = window_size, 
              diff_price = close, 
              batch_size = batch_size)
RL_Agent.train(episodes = 150 , checkpoint = 1, initial_capital = initial_capital)

states_buy, states_sell, total_invest_perf, total_account_gains = RL_Agent.positions(initial_capital_p = initial_capital)

fig = plt.figure(figsize = (25,12))
plt.plot(close, color='r', lw=2.)
plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
plt.title('total gains %.3f, total investment performance%.3f%%'%(total_account_gains, total_invest_perf))
plt.legend()
plt.show()
