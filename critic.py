# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:14:30 2017

@author: Gergo
"""
#%%
import numpy as np
import math
from keras.initializers import normal, identity
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers import Dense, Add, Concatenate,BatchNormalization,Dropout, Flatten, Input, merge, Lambda, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

HIDDEN_1 = 80
HIDDEN_2 = 40

class Critic:
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LR):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LR = LR
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state_1,self.state_2 = self.create_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_state_1,self.target_state_2 = self.create_network(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state_1: np.array(states[:,0:2]),
            self.state_2: np.array(states[:,2:4]),
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_network(self, state_size,action_dim):
        print("Now we build the model")
        inp_1 = Input(shape=[state_size-2],name='state_1')
        inp_2 = Input(shape=[2],name='state_2')
        A = Input(shape=[action_dim],name='action2')
          
        
        h00 = Dense(HIDDEN_1, activation='relu')(inp_1)
        h01 = Dense(HIDDEN_1, activation='relu')(h00)
        h02 = Dense(HIDDEN_2, activation='relu')(h01)
        
        h10 = Dense(HIDDEN_1, activation = 'relu')(inp_2)
        h11 = Dense(HIDDEN_2, activation = 'relu')(h10)
        h2 =  Concatenate()([h02,h11]) 
        h3 = Dense(HIDDEN_2, activation='relu')(h2)
        
        a1 = Dense(HIDDEN_2, activation='linear')(A) 
        h4 = Add()([h3,a1])         
        h5 = Dense(HIDDEN_2, activation='relu')(h4)
        V = Dense(action_dim,activation='linear')(h5)   
        model = Model(inputs=[inp_1,inp_2,A],outputs=V)
        adam = Adam(lr=self.LR)
        model.compile(loss='mse', optimizer=adam)
        return model, A, inp_1,inp_2 