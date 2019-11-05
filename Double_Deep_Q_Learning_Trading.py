
# Double Deep Q-Learning


import time

import tensorflow as tf
from keras import backend as K
K.tensorflow_backend._get_available_gpus()

import keras
config = tf.ConfigProto( device_count = {'GPU': 0} ) 
sess = tf.Session(config=config) 
keras.backend.set_session(sess)

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set()

plt.style.use('ggplot')

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop

from keras.initializers import glorot_normal
from keras import regularizers

import pandas_datareader.data as web
df = web.DataReader('JPM',data_source="yahoo",start="03/01/1998",end="30/6/2010").dropna()  # JP Morgan stock 

#df.Close.plot()

print(df.head())
print(df.shape)

#rounding dataframe for even more discrete states

df = df.round(2)
print(df.head())
print('\n', df.tail())

from collections import deque
import random

class DDQN_Agent:
  
    def __init__(self, state_size, window_size, price, batch_size, ): 
      
      
        self.state_size = state_size
        self.window_size = window_size
        self.price = price
        self.action_size = 3
        self.batch_size = batch_size
        self.memory = deque(maxlen = 10000)
        #TODO: add commisions
        
        
        
        self.gamma = 0.5 
        self.epsilon = 0.8 
        self.epsilon_min = 0.1 # relative high value in order to continue to learn even in later episodes #was 0.01
        self.epsilon_decay = 0.997 

        
        self.optim = Adam(lr = 1e-2) #was 1e-3
        
        #Q model
        self.QNN = self.brain()
        self.QNN.compile(loss= tf.losses.huber_loss, metrics = ['mean_squared_error'], optimizer= self.optim)
        
        #Q target model
        self.Target_QNN = self.brain()
        self.counter = 0
        self.update_target_weights()
       
        
    def brain(self):
        model = Sequential()
        
        
        model.add(Dense(128, input_dim= self.state_size, activation='relu',
                       kernel_initializer = 'glorot_normal', kernel_regularizer= regularizers.l2(1e-6)))
        
        
        model.add(Dense(32,activation='relu',
                        kernel_initializer = 'glorot_normal', kernel_regularizer= regularizers.l2(1e-6)))
        
        model.add(Dense(8, activation='relu', kernel_initializer = 'glorot_normal'))
        model.add(Dense(self.action_size,activation='linear'))
        
        
        
        model.summary()
        return model
      
      
    def target_Qvalue(self, next_state):
      
      #Bellman equation for Double Q-learning
      
      #QNN chooses action...
      actionQNN = np.argmax(self.QNN.predict(next_state)[0]) #an integer (essentially showing the position of the chosen action for that state)
      # ... Target_QNN evaluates it
      Qvalue_next_state = self.Target_QNN.predict(next_state)[0][actionQNN]
      
      Qvalue = self.gamma*Qvalue_next_state
      
      return Qvalue
      
      
      
      
      
    def update_target_weights(self):
        self.Target_QNN.set_weights(self.QNN.get_weights())
        
        
    def act(self, state): #following an espilon-greedy policy with probability of exploitation [1-epsilon] and exploration [epsilon]
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
                  
        return np.argmax(self.QNN.predict(state)[0]) # keras model.predict() outputs in the form array([[-0.82]], that's why we use the [0] to access 
                  

    # Reminder: In every function only one "return" is returned
    
   
    #The sigmoid function won't be used, but is defined in case further normalization of the state vector is required
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))
      
      
    def get_state(self, t):
        window_size = self.window_size
        d = t - window_size
        
        if d>=0:
          window = self.price[d:t+1]
          #print(window,len(window)) #for debugging

        else:
          d= -d
          padding = d *[self.price[0]]
          window = padding + self.price[0 : t + 1]
          
          #print("Dbg: else", window,len(window)) #for debugging
        
        state_series = []
        for i in range(window_size): # an integer/length of the window
            #The number of the following differences will be window_size - 1 . E.g. I have a vector of 11 values, the vector of their differences will be 10
            state_series.append(np.log(window[i + 1]/window[i])*100) # calculates a simple first differences for the closing prices in the window: closing_price[t] - closing_price[t-1]
                                                                         # was -> state_series.append(self.sigmoid(window[i + 1] - window[i])) 
        #print(window, len(window))  #for debugging

        return np.array([state_series])
      
      
    
    def sample_data(self, x, size):
        return random.sample(list(x), size) #  random.sample(sequence, k) -> samples k size from sequence
             
              
      
    def experience_replay(self, batch_size, random_sampling= False):
        
        mini_batch = []
        
        if random_sampling is False:
            l = len(self.memory)
            for i in range(l - batch_size, l):
                mini_batch.append(self.memory[i])
        else:
            mini_batch = [i for i in self.sample_data(self.memory, batch_size)]
        
        assert len(mini_batch) == batch_size
        exp_replay_size = len(mini_batch)
        
        buffer_states = np.empty((exp_replay_size, self.state_size))
        buffer_next_states = np.empty((exp_replay_size, self.state_size))
        buffer_Qvalues = np.empty((exp_replay_size, self.action_size))
        
        states = np.array([s[0][0] for s in mini_batch])
        next_states = np.array([s[3][0] for s in mini_batch])
        
        Q = self.QNN.predict(states)
        
        
        
        for i in range(len(mini_batch)):
          
            state, action, reward, next_state, done = mini_batch[i]
            
            
            target = Q[i]
            target[action] = reward
            
            if not done: # WILL EXECUTE if done == False
                
                Q_target_q_value = self.target_Qvalue(next_state)
                target[action] += Q_target_q_value 
            
            buffer_states[i] = state # When fitted in the model.fit() this input will give Q and the error will be computed between Q(bellman)* and the former Q
            buffer_Qvalues[i] = target
        
     
        history = keras.callbacks.History()
        self.QNN.fit(buffer_states,buffer_Qvalues, epochs=1, verbose=0, callbacks = [history])
        online_train_loss = history.history['loss'][0]
                      
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
            #print(self.epsilon) #for debugging
          
        if self.counter % 10 == 0: #was 5
            self.update_target_weights()
        
        self.counter+= 1
        
        return online_train_loss
    
         
    
          
    def train(self, episodes, checkpoint, initial_capital):
      
        for i in range(episodes):
            total_profit = 0
            portfolio = []
            
            state = self.get_state(0) #Initializing the state variable
            
            present_capital = initial_capital
            total_investment = 0
            trading_reward = 0
            rewards_list = []
            
            start = time.time()
            for t in range(0, len(self.price)): 
                
                
                action = self.act(state)
                
                if t == len(self.price)-1 :
                    continue
                else:
                    next_state = self.get_state(t + 1)
                
                if action == 1 and present_capital >= self.price[t]: 
                    portfolio.append(self.price[t])                                                              
                    present_capital -= self.price[t]
                    #total_profit -= self.commission_fee
                    #present_capital += total_profit
                    if t < len(self.price)-1:
                      if self.price[t] > self.price[t+1]:
                        trading_reward = -1 
                    else:
                      trading_reward = 0
                    #rewards_list.append(trading_reward)
                    
                elif action == 2 and len(portfolio) > 0:
                    bought_price = portfolio.pop(0)
                    difference = self.price[t] - bought_price
                    total_profit += difference
                    #total_profit -= self.commission_fee
                    present_capital += self.price[t]
                   
                    
                    trading_reward = difference+1 if difference > 0 else difference-2 #max(total_profit,0)
                    
                else:
                    if t < len(self.price)-1 and len(portfolio) > 0:
                      trading_reward = -1/2 if self.price[t] > self.price[t+1] else 1/2
                    else:
                      trading_reward = 0
                
                rewards_list.append(trading_reward)
      
      
        
        
                if t == len(self.price) - 1:
                    done = True
                    if len(portfolio) > 0:
                      for i in portfolio:
                        present_capital += i 
                elif present_capital < 0 or present_capital == 0:
                    done = True
                else:
                    done = False
                
                self.memory.append((state, action, trading_reward, 
                                    next_state, done))
                
                batch_size = min(self.batch_size, len(self.memory))
                loss = self.experience_replay(batch_size, random_sampling= False ) 
                
                if done == True:
                    if t < len(self.price)-1:
                      print('*** Final state was reached *before* the last price-state (!!!) ***')
                      break             
                state = next_state
                
            total_investment = ((present_capital - initial_capital) / initial_capital)*100 
            rewards_mean = np.asarray(rewards_list).mean()
                
            end = time.time()
            total_time = end - start
            print('Estimated time (minutes) to run this episode %.2f'%(total_time/60))
                
            if (i+1) % checkpoint == 0:
              
              print('-------------------------------------------------------------------------------------------------------------------------------')
              print('episode:', i+1, ' -- ', 'online_loss: ', round(loss,12), ' -- ', '\n')                    
                    
              print('mean of rewards: %.4f, total profit: %.2f, investment performance: %.2f %%, total capital: %.3f, portfolio status: %s'
                     %(rewards_mean, total_profit, total_investment, present_capital, portfolio))
                                                                               
              print('\n', 'memory length is {}'.format(len(self.memory)))
              
              print('-------------------------------------------------------------------------------------------------------------------------------', '\n')
              
              
              
              
    def positions(self, initial_capital_p):
        initial_capital_p = initial_capital_p
        present_capital_p = initial_capital_p
        states_sell = []
        states_buy = []
        portfolio_p = []
        total_profit_p = 0
        state = self.get_state(0)
        difference = 0
        
        for t in range(0, len(self.price) - 1): 
            action = self.act(state)
            next_state = self.get_state(t + 1)
            
            if action == 1 and present_capital_p >= self.price[t]: #and t < (len(self.price) -10): 
                    portfolio_p.append(self.price[t])
                    present_capital_p -= self.price[t]
                    states_buy.append(t)
                    print('Day %d: Agent buys 1 unit at price %f, total account balance %f'% (t, self.price[t], present_capital_p))
                
                
            elif action == 2 and len(portfolio_p)>0:
                    bought_price = portfolio_p.pop(0)
                    difference = self.price[t] - bought_price
                    total_profit_p += difference
                    present_capital_p += self.price[t]
                    states_sell.append(t)
                    
                    investment_performance = ( difference / bought_price) * 100
                    print('Day %d, Agent sells 1 unit at price %f, total profit: %.2f, investment performance %f %%, total account capital %f,'
                           %(t, self.price[t], total_profit_p, investment_performance, present_capital_p))
            
            state = next_state
         
        if len(portfolio_p) > 0:
          for p in portfolio_p:
            present_capital_p += p
        total_invest_perf = ((present_capital_p - initial_capital_p) / initial_capital_p) * 100
        total_account_gains = present_capital_p - initial_capital_p
        return states_buy, states_sell, total_invest_perf, total_account_gains

close = df.Close.values.tolist()
initial_capital = 10000 
window_size = 10
batch_size = 32 
RL_Agent = DDQN_Agent(state_size = window_size, 
              window_size = window_size, 
              price = close, 
              batch_size = batch_size)
RL_Agent.train(episodes = 120, checkpoint = 10, initial_capital = initial_capital)

states_buy, states_sell, total_invest_perf, total_account_gains = RL_Agent.positions(initial_capital_p = initial_capital   )

fig = plt.figure(figsize = (25,9))
plt.plot(close, color='k', lw=2.)
plt.plot(close, '^', markersize=10, color='g', label = 'buying signal', markevery = states_buy)
plt.plot(close, 'v', markersize=10, color='r', label = 'selling signal', markevery = states_sell)
plt.title('total gains %.3f, total investment performance %.3f%%'%(total_account_gains, total_invest_perf))
plt.legend()
plt.show()


