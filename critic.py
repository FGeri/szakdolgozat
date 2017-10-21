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
from keras.layers import Dense, Add, Flatten, Input, merge, Lambda, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

HIDDEN_1 = 75
HIDDEN_2 = 150

class Critic:
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LR):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LR = LR
        self.action_size = action_size
        
        K.set_session(sess)

        #Now create the model
        self.model, self.action, self.state = self.create_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_state = self.create_network(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
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
        S = Input(shape=[state_size])  
        A = Input(shape=[action_dim],name='action2')   
        w1 = Dense(HIDDEN_1, activation='relu')(S)
        a1 = Dense(HIDDEN_2, activation='linear')(A) 
        h1 = Dense(HIDDEN_2, activation='linear')(w1)
        h2 = Add()([h1,a1])    
        h3 = Dense(HIDDEN_2, activation='relu')(h2)
        V = Dense(action_dim,activation='linear')(h3)   
        model = Model(inputs=[S,A],outputs=V)
        adam = Adam(lr=self.LR)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S 