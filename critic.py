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
from keras.optimizers import Adam,Nadam,RMSprop
import keras.backend as K
import tensorflow as tf

HIDDEN_1 = 80
HIDDEN_2 = 40

class Critic:
    def __init__(self, sess, mode, state_size, action_size, BATCH_SIZE, LR,output_size, structure):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.LR = LR
        self.action_size = action_size
        self.mode = mode
        self.structure = structure
        
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

    def target_train(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def _huber_loss(self, target, prediction):
        error = prediction - target
        return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)
    
    def create_network(self, state_size,action_dim,output_size):
        if self.mode =="LIDAR":
            print("Creating Critic NN - LIDAR mode")
            inp_1 = Input(shape=[state_size-2,1],name='state_1')
            inp_2 = Input(shape=[2],name='state_2')
#            Branch 1
            if self.structure == "SIMPLE_DENSE":
                flat_19 = Flatten()(inp_1)
            elif self.structure == "1D_CONVOLUTION":
                conv_10 = Conv1D(filters=16,kernel_size=4,strides=1,activation="relu")(inp_1)
                conv_11 = Conv1D(filters=16,kernel_size=3,strides=1,activation="relu")(conv_10)
                conv_13 = Conv1D(filters=8,kernel_size=3,strides=1,activation="relu")(conv_11)
                pool_14 = MaxPool1D(pool_size=2)(conv_13)
                flat_19 = Flatten()(pool_14)
                
##            Branch 2
#            dense_20 = Dense(128)(inp_2)
#            Branch 1+2
            merge_40 = Concatenate()([flat_19,inp_2])
            dense_41 = Dense(64, activation='relu')(merge_40)
            dense_42 = Dense(32, activation='relu')(dense_41)
            dense_43 = Dense(32, activation='relu')(dense_42)
            V = Dense(output_size,activation='linear')(dense_43) 
            model = Model(inputs=[inp_1,inp_2],outputs=V)
            optimizer = Adam(lr=self.LR)
            model.compile(loss='mse', optimizer=optimizer)
        return model, inp_1,inp_2 