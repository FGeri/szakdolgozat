# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 13:14:30 2017

@author: Gergo
"""
#%%
import numpy as np
import math
import keras
from keras.initializers import normal, identity
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers import Dense, Add, Concatenate,BatchNormalization,Dropout
from keras.layers import Flatten, Input, merge, Lambda, Activation
from keras.layers import Conv1D,MaxPool1D,LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam,SGD,RMSprop
import keras.backend as K
import tensorflow as tf

HIDDEN_1 = 80
HIDDEN_2 = 40

class Critic:
    def __init__(self, sess, mode, state_size, action_size, BATCH_SIZE, TAU, LR,output_size):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LR = LR
        self.action_size = action_size
        self.mode = mode
        
        K.set_learning_phase(1)
        K.set_session(sess)

        #Now create the model
        self.model, self.state_1,self.state_2 = self.create_network(state_size, action_size,output_size)  
        self.target_model, self.target_state_1,self.target_state_2 = self.create_network(state_size, action_size,output_size)  
#        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        self.sess.run(tf.global_variables_initializer())

    def gradients(self, states, actions):
        if self.mode == "LIDAR":
            return self.sess.run(self.action_grads, feed_dict={
                self.state_1: np.expand_dims(np.array(states[:,0:-2]),axis=2),
                self.state_2: np.array(states[:,-2:]).reshape([-1,2])
            })[0]
        else:
            return self.sess.run(self.action_grads, feed_dict={
                self.state_1: np.array(states[:,0:2]),
                self.state_2: np.array(states[:,2:4])
            })[0]

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())

    def create_network(self, state_size,action_dim,output_size):
        if self.mode =="LIDAR":
            print("Creating Critic NN - LIDAR mode")
            inp_1 = Input(shape=[state_size-3,1],name='state_1')
            inp_2 = Input(shape=[3],name='state_2')
#            Branch 1
#            conv_10 = Conv1D(filters=32,kernel_size=3,strides=1,activation="relu")(inp_1)
#            conv_11 = Conv1D(filters=16,kernel_size=3,strides=1,activation="relu")(conv_10)
##            norm_14 = BatchNormalization()(conv_13)
#            pool_12 = MaxPool1D(pool_size=2)(conv_11)
#            conv_13 = Conv1D(filters=16,kernel_size=3,strides=1,activation="relu")(pool_12)
##            norm_17 = BatchNormalization()(conv_16)
#            pool_14 = MaxPool1D(pool_size=2)(conv_13)
            flat_19 = Flatten()(inp_1)
            
##            Branch 2
#            dense_20 = Dense(128)(inp_2)
#            Branch 1+2
            merge_40 = Concatenate()([flat_19,inp_2])
            dense_41 = Dense(64)(merge_40)
#            dense_41 = BatchNormalization()(dense_41)
            dense_41 = Activation('relu')(dense_41)
            dense_42 = Dense(32)(dense_41)
#            dense_42 = BatchNormalization()(dense_42)
            dense_42 = Activation('relu')(dense_42)
            dense_43 = Dense(32, activation='relu')(dense_42)
#            dense_44 = Dense(30, activation='relu')(dense_43)
            V = Dense(output_size,activation='linear')(dense_43) 
            model = Model(inputs=[inp_1,inp_2],outputs=V)
            optimizer = Adam(lr=self.LR)
#            optimizer = RMSprop(lr=self.LR, epsilon=0.01, decay=0.95)
            model.compile(loss='mse', optimizer=optimizer)
            
        else:
            print("Creating Critic NN - GLOBAL mode")            
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
            V = Dense(1,activation='linear')(h5)   
            model = Model(inputs=[inp_1,inp_2,A],outputs=V)
#            optimizer = Adam(lr=self.LR)
            optimizer = RMSprop(lr=self.LR, epsilon=0.01, decay=0.95)
            model.compile(loss='mse', optimizer=optimizer)
        return model, inp_1,inp_2 