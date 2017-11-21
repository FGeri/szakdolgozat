# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 14:02:38 2017

@author: Gergo
"""
import pickle
import keras
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation,BatchNormalization
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import plot_model

def softmax(a):
    x = np.asarray(a,dtype=float)
    x -= np.max(x)
    exps = np.exp(x)
    return  exps/exps.sum()

def pick_action(state,strategy = None):
    s = np.zeros([4])
    s[0] = np.argmax(state[0,0*10:1*10])
    s[1] = np.argmax(state[0,1*10:2*10])
    s[2] = np.argmax(state[0,2*10:3*10])
    s[3] = np.argmax(state[0,3*10:4*10])
    if not strategy:
        a = (int(s[3]/3)+2)%3 if s[3]/3 <3 else 0
        if not(0 in [int(s[0]/3),int(s[1]/3)]):
            a = 0
        if not(1 in [int(s[0]/3),int(s[1]/3)]):
            a = 1
        if not(2 in [int(s[0]/3),int(s[1]/3)]):
            a = 2
        if s[1]/3 >=3:
            a = 0
#        a = np.random.random_integers(0,2)
    elif strategy =="LAST":
        a = (int(s[0]/3)+1)%3 if s[0]/3 <3 else 0 
    else:
        a = (int(s[0]/3)+1)%3 if s[0]/3 <3 else 0 
    return a
    
    
np.random.seed(123)

#Hyper parameters
EPISODES = 300
MAX_STEPS = 40
EXPLORE = 200

LEARNING_RATE = 0.01
TARGET_UPDATE_FREQ = 1
TARGET_UPDATE_FREQ_EXP = 1.2

BATCH_SIZE = 32
GAMMA = 0.99

epsilon = 2   
n = 4

states = ["kő-kő",          #0
          "kő-papír",       #1
          "kő-olló",        #2
          "papír-kő",       #3
          "papír-papír",    #4
          "papír-olló",     #5
          "olló-kő",        #6
          "olló-papír",     #7
          "olló-olló",      #8
          "N/A"]            #9


actions = [0,1,2] # Kő , papír, olló
actions = np.reshape(actions,[-1,1])
states = np.array(states)

state_encoder = OneHotEncoder(sparse = False)
state_encoder.fit(np.reshape(np.array(range(10)),[-1,1]))

action_onehot_encoder = OneHotEncoder(sparse=False)
action_onehot_encoder.fit(actions)


#Creating the network
model = Sequential()
model.add(Dense(units = 40,activation='relu',input_dim = 10*n))
model.add(Dense(units= 10,activation='relu'))
model.add(Dense(units= 3,activation='linear'))
model.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
                loss='mse')

#Creating the target network
target = Sequential()
target.add(Dense(units = 40,activation='relu',input_dim = 10*n))
target.add(Dense(units= 10,activation='relu'))
target.add(Dense(units= 3,activation='linear'))
target.compile(optimizer=keras.optimizers.Adam(lr=LEARNING_RATE),
                loss='mse')
target.set_weights(model.get_weights())

#Create experience replay memory
buffer = pd.DataFrame([[np.array([]),np.array([]),0.,np.array([]),False]],columns=['s', 'a', 'r', "s_", 'terminal'])
buffer = buffer.drop(buffer.index[0])

#Generate training data

total_steps = 0
log = []
average_q_values = []
for i in range(EPISODES):
    total_reward = 0 
    state = np.hstack([state_encoder.transform(np.reshape(np.asarray([9]),[1,1])),
                      state_encoder.transform(np.reshape(np.asarray([9]),[1,1])),
                      state_encoder.transform(np.reshape(np.asarray([9]),[1,1])),
                      state_encoder.transform(np.reshape(np.asarray([9]),[1,1]))])
    for step in range(MAX_STEPS):
        total_steps +=  1
        epsilon -= 1/EXPLORE
#        Agent's action
        y_pred = model.predict(state)
        average_q_values.append(np.mean(y_pred))
        p = softmax(y_pred.reshape(-1)/max(epsilon,0.001))
        a = np.random.choice(range(3),1,p=p)
        
#        Opponent's actions
        opp_a = pick_action(state)
        
#        Observe next state
        state_ = np.hstack([state_encoder.transform(np.reshape(np.asarray([a*3+opp_a]),[1,1])),
                      np.atleast_2d(state[0,10:])])
        r = 1 if a == (opp_a+1)%3 else -1
        terminal = True if step == 4 else False
        
        total_reward += r
#        Store experience in memory
        experience =  pd.DataFrame([[state.reshape(-1),a,r,state_.reshape(-1),terminal]],columns=['s', 'a', 'r', "s_", 'terminal'],copy = True)
        buffer = pd.concat([buffer,experience],ignore_index=True)
        
#        Update state
        state = state_
#        Sample mini batch from replay memory
        if buffer.shape[0] < BATCH_SIZE:
            batch = buffer.values[:]
        else:
            batch = buffer.sample(BATCH_SIZE,replace=False).values[:]
            
        states = np.asarray(np.atleast_2d([e[0] for e in batch]))
        actions = np.asarray([e[1] for e in batch])
        actions = actions.reshape(-1)
        rewards = np.asarray([e[2] for e in batch])
        states_ = np.asarray(np.atleast_2d([e[3] for e in batch]))
        terminals = np.asarray([e[4] for e in batch])

#        Train network    
        y_preds = model.predict_on_batch([states])
        y_next_preds = model.predict_on_batch([states_])
        next_a_indeces = np.argmax(y_next_preds,axis=1)
        maxQs = target.predict_on_batch([states_])
        maxQs = maxQs[range(len(batch)),next_a_indeces]
        targets = rewards + GAMMA*maxQs*(1-terminals)
        
#        a_indeces = []
#        for row in actions:
#            a_indeces.append(int(np.where(np.all(sampled_actions==row,axis=1))[0]))
#            errors = y[range(len(batch)),a_indeces] - targets
#            buff.update_priorities(indeces,errors)
        
        y_preds[range(len(batch)),actions] = targets
        model.fit([states], y_preds,epochs=1, verbose=0)       
        
#        Update target network
        if total_steps % TARGET_UPDATE_FREQ == 0 :
            target.set_weights(model.get_weights())
            TARGET_UPDATE_FREQ =int(TARGET_UPDATE_FREQ * TARGET_UPDATE_FREQ_EXP)
    log.append(total_reward)
    print("Episode: "+str(i)+"\tTotal reward: "+str(total_reward))
    
with open("ddqn_q_values.txt", "wb") as fp:
    pickle.dump(average_q_values, fp)    
fig = plt.figure()
plt.plot((MAX_STEPS+np.asarray(log))/2/MAX_STEPS*100)
fig.suptitle('DQN', fontsize=20)
plt.xlabel('Episodes', fontsize=18)
plt.ylabel('Won games (%)', fontsize=16)
fig.savefig('DQN_kopapirollo_python_target_nelkul.jpg')

#plot = plt.figure()    
#plt.plot(np.asarray(range(len(log))),np.asarray(log))    