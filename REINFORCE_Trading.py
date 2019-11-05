
# Monte-Carlo Policy Gradient (REINFORCE)

from keras.layers import Input, Dense
from keras.models import Model



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
from keras.utils.np_utils import to_categorical

from collections import deque
import random

import pandas_datareader.data as web
df = web.DataReader('IBM',data_source="yahoo",start="08/01/2017",end="06/01/2018").dropna()  # IBM stock 

print(df.head())
print(df.shape)

df = df.round(2)
df = df.dropna()



class REINFORCEAgent:
    
    
    def __init__(self, state_size, window_size, price, hidden1_nodes, hidden2_nodes):
      
      
      self.state_size = state_size
      self.window_size = window_size
      self.action_size = 3
      self.price = price
      self.hidden1_nodes = hidden1_nodes
      self.hidden2_nodes = hidden2_nodes
      
      
      self.policy_n_network_(hidden1 = hidden1_nodes, hidden2 = hidden2_nodes)
      
      self.policy_training()
      
      self.gamma = 0.6
    
    
      self.policy_n_network.summary()
    
    # approximate policy using Neural Network
    # state is input and probability of each action is output of network
    def policy_n_network_(self, hidden1, hidden2):
      
      
      inputs = Input(shape=(self.state_size,))

      layer1 = Dense(self.hidden1_nodes, activation='tanh', kernel_initializer='glorot_normal')(inputs)
      layer2 = Dense(self.hidden2_nodes, activation='sigmoid', kernel_initializer='glorot_normal')(layer1)
      outputs = Dense(self.action_size, activation='softmax')(layer2)
        
      self.policy_n_network = Model(inputs,outputs)
      
     
    def action(self,state):
      action_probs = self.policy_n_network.predict(state, batch_size=1).flatten()
      #print(action_probs) #for debugging
      action = np.random.choice(self.action_size, 1, p=action_probs)[0]
      return action

      
    def disc_rewards(self, rewards):
   
      rewards_len = len(rewards)
      discounted_rewards_list = []
      
      for t in range(rewards_len):
        
        
        gamma = self.gamma
        gamma_t = 1
        discounted_rewards_t = rewards[t]
        
        for reward_iter in rewards[t+1:]:
                
          discounted_rewards_t += gamma ** gamma_t *reward_iter #discounted_rewards_t + gamma ** gamma_t * rewards[t]
          gamma_t += 1
         
        discounted_rewards_list.append(discounted_rewards_t)
        
      baseline = K.mean(K.variable(discounted_rewards_list),axis=-1)
      
      d_rewards = K.variable(discounted_rewards_list) - baseline
      
      return d_rewards
      
      
      
      
    def policy_training(self):                                                    # * K.function( ) effectively RUNS THE GRAPH, giving us what we require by following
                                                                                  # the GRAPH and outputting accordingly (!) *   
      
      action_softmax = self.policy_n_network.output
      actions_one_hot_holder = K.placeholder(shape = (None, self.policy_n_network.layers[-1].output_shape[1]))
      disc_reward_holder = K.placeholder(shape = (None, ))
      
      
      
      reward_x_one_hot = tf.multiply(disc_reward_holder, actions_one_hot_holder) # K.stop_gradient(disc_reward_holder): for explicitly stating
      REINFORCE_loss = - K.sum(tf.multiply(K.log(action_softmax), reward_x_one_hot), axis= -1)
      
      
                                                                                                        # gradient descend on: - ((Reward) * (log(policy(θ|states))))
                                                                                                        # .. effectively maximizing for the log probabilities of the
                                        
      #RMSprop_opt = keras.optimizers.RMSprop()                                                         # specific action taken in that state, if and only if the
      Adam_opt = keras.optimizers.Adam()                                                                # the discounted reward is positive. On the other hand, if it
                                                                                                        # negative, we will "push down"/minimize the probabilities   
      weight_updates = Adam_opt.get_updates(params=self.policy_n_network.trainable_weights,             # for that action                     
                                   loss = REINFORCE_loss)

      self.REINFORCE_train_function = K.function(inputs=[self.policy_n_network.input,
                                           actions_one_hot_holder,
                                           disc_reward_holder, K.learning_phase()],
                                   outputs= [],
                                   updates= weight_updates)                                             # weight updates for minimizing the loss (which is given in
                                                                                                        # another function, "optimizer".get_updates() and is masked
                                                                                                        # by the variable "weight_updates" here. In order to actually
                                                                                                        # train, we need the neural network inputs for our custom loss,
                                                                                                        # which in term will be passed to the weight updating process.
                                                                                                        
    def policy_update(self, states, actions, rewards):
                                        
      a_one_hot = to_categorical(actions, self.policy_n_network.layers[-1].output_shape[1]) 
      #print(a_one_hot) #for debugging
      get_disc_reward  = self.disc_rewards(rewards)
                                        
      s = states                                  
      r = get_disc_reward
      
            
      for i in range(len(states)-1):
        s_ = states[i]
        a_one_hot_ = a_one_hot[i]
        r_ = r[i]
        self.REINFORCE_train_function([s_, a_one_hot_, r_])

                                        
                                        
                                        
                                        
    def get_state(self, t):
        window_size = self.window_size
        d = t - window_size
        
        if d>=0:
          window = self.price[d:t+1]
          #print(window,len(window)) #for debugging

        else:
          d= -d
          padding = d *[self.price[0] + 1e-10]
          window = padding + self.price[0 : t + 1]
          
          #print('Dbg: the else print', window, len(window)) #for debugging
        
        state_series = []
        for i in range(window_size): # an integer/length of the window
            #The number of the following differences will be window_size - 1 . E.g. I have a vector of 11 values, the vector of their differences will be 10
            state_series.append(np.log(window[i + 1]/window[i])*100) 
                                                                         # was -> state_series.append(self.sigmoid(window[i + 1] - window[i])) 
        #print(window, len(window))  #for debugging

        return np.array([state_series])                                        
            
      
                                      
                                        
                                        
    def train(self, episodes, checkpoint, initial_capital):
      
        for i in range(episodes):
            total_profit = 0
            portfolio = []
            
            
            present_capital = initial_capital
            total_investment = 0
            trading_reward = 0
            
            states_list = []
            
            actions_list = []
            rewards_list = []
            state = self.get_state(10)
            states_list.append(state)
            
            done = False
                                        
            start = time.time()
            
            #print(states_list) #for debugging
                                        
            for t in range(11, len(self.price)-1): 
                
                
                action = self.action(state)
                states_list.append(state)
                
                #print(states_list) #for debugging
                #print(action) #for debugging
                
                actions_list.append(action)
                
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
                        trading_reward = -1.5 
                      else:
                        trading_reward = 1.5
                    #rewards_list.append(trading_reward)
                    
                elif action == 2 and len(portfolio) > 0:
                    bought_price = portfolio.pop(0)
                    difference = self.price[t] - bought_price
                    total_profit += difference
                    #total_profit -= self.commission_fee
                    present_capital += self.price[t]
                    
                    trading_reward = difference+2 if difference > 0 else difference -2.5
                   
                elif action == 2 and len(portfolio) == 0:
                    trading_reward = -1.5 if self.price[t] < self.price[t+1] else 1.5 
                    
                else:
                    if t < len(self.price)-1 and len(portfolio) > 0:
                      trading_reward = -1.1 if self.price[t] > self.price[t+1] else 1.1
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
                
                
                
                if done == True:
                    if t < len(self.price)-2:
                      print('*** Final state was reached *before* the last price-state (!!!) ***')
                      break  
                                        
                                        
                state = next_state
                #print(states_list) #for debugging
                
            total_investment = ((present_capital - initial_capital) / initial_capital)*100 
            rewards_mean = np.asarray(rewards_list).mean()
                

                                        
            # UPDATING the Policy Neural Network
            self.policy_update(np.array(states_list), np.array(actions_list), np.array(rewards_list))
            
            end = time.time()
            total_time = end - start
            print('Estimated time (minutes) to run this episode %.2f'%(total_time/60))
                                        
            if (i+1) % checkpoint == 0:
              
              print('\n')
              print('-------------------------------------------------------------------------------------------------------------------------------')
              print('episode:', i+1, ' -- ', '\n')
              print('\n', 'The (undiscounted) for the episode are {}'.format(rewards_list), '\n')
              print('mean of rewards: %.4f, total profit: %.2f, investment performance: %.2f %%, total capital: %.3f, portfolio status: %s'
                     %(rewards_mean, total_profit, total_investment, present_capital, portfolio))
              print('\n', 'The states list length is: {}'.format(len(states_list)))
              print('-------------------------------------------------------------------------------------------------------------------------------', '\n')
              print('\n')
              
              
              
    def positions(self, initial_capital_p):
        initial_capital_p = initial_capital_p
        present_capital_p = initial_capital_p
        states_sell = []
        states_buy = []
        portfolio_p = []
        total_profit_p = 0
        state = self.get_state(10)
        difference = 0
        
        for t in range(11, len(self.price) - 1): 
            action = self.action(state)
            next_state = self.get_state(t + 1)
            
            if action == 1 and present_capital_p >= self.price[t]: #and t < (len(self.price) -10): # - self.half_window):
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
initial_capital = 1e6 
window_size = 10
RL_Agent = REINFORCEAgent(state_size = window_size, 
              window_size = window_size, 
              price = close,
              hidden1_nodes = 32,
              hidden2_nodes = 8)
             
             
RL_Agent.train(episodes = 15, checkpoint = 1, initial_capital = initial_capital)

#AFTER 15 EPISODES ( [!] : 15 is NOT a sufficient number of episodes for this algorithm.  )
states_buy, states_sell, total_invest_perf, total_account_gains = RL_Agent.positions(initial_capital_p = initial_capital)


fig = plt.figure(figsize = (15,8))
plt.plot(close, color='k', lw=1.5)
plt.plot(close, '^', markersize=10, color='g', label = 'buying signal', markevery = states_buy)
plt.plot(close, 'v', markersize=10, color='r', label = 'selling signal', markevery = states_sell)
plt.title('total gains: %.3f, total investment performance: %.3f%%'%(total_account_gains, total_invest_perf))
plt.legend()
plt.show()



#Alternative graph

fig = plt.figure(figsize = (15,8))
plt.plot(close, color='k', lw=1.5)
plt.plot(close, '^', markersize=10, color='g', label = 'buying signal', markevery = states_buy)
plt.plot(close, 'v', markersize=10, color='r', label = 'selling signal', markevery = states_sell)
plt.title(' A.I. Trading Bot 3: Monte Carlo Policy Gradient - REINFORCE with baseline')
plt.ylabel('IBM stock price')
plt.xlabel('Observations')
plt.legend(framealpha=1, shadow=True, borderpad=1)
plt.show()


