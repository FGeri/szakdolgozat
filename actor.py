# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:46:11 2017

@author: Gergo
"""
#%%
from keras.models import Sequential,Model,model_from_json
from keras.initializers import normal
from keras.layers import Dense, Input, Concatenate
from keras.layers import Lambda,Activation,BatchNormalization,Dropout 
from keras.optimizers import Adam
import keras
import numpy as np
import tensorflow as tf
import keras.backend as K


HIDDEN_1 = 75
HIDDEN_2 = 150


class Actor:
    def __init__(self, sess, state_size, action_size, BATCH_SIZE, TAU, LR):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LR = LR

        K.set_session(sess)
        self.model , self.weights, self.state = self.create_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LR).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())
    
    
    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })
    
    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)
        
    def create_network(self,state_size,action_dim):
        S = Input(shape=[state_size])   
        h0 = Dense(HIDDEN_1, activation='relu')(S)
#        TODO Add batchnormalisation
#        TODO ADd dropout?
        h1 = Dense(HIDDEN_2, activation='relu')(h0)
        Acceleration = Dense(1,activation='tanh')(h1)
        Steering = Dense(1,activation='tanh')(h1)        
        V = Concatenate()([Steering,Acceleration])          
        model = Model(inputs=S,outputs=V)
        return model, model.trainable_weights, S

