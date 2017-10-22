# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 12:46:11 2017

@author: Gergo
"""
#%%
from keras.models import Sequential,Model,model_from_json
from keras.initializers import normal
from keras.layers import Dense, Input, Concatenate,Conv1D,MaxPool1D,Flatten
from keras.layers import Lambda,Activation,BatchNormalization,Dropout 
from keras.optimizers import Adam
import keras
import numpy as np
import tensorflow as tf
import keras.backend as K


HIDDEN_1 = 80
HIDDEN_2 = 40


class Actor:
    def __init__(self, sess, mode, state_size, action_size, BATCH_SIZE, TAU, LR):
        self.sess = sess
        self.BATCH_SIZE = BATCH_SIZE
        self.TAU = TAU
        self.LR = LR
        self.mode = mode

        K.set_session(sess)
        self.model , self.weights, self.state_1,self.state_2 = self.create_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state_1,self.target_state_2 = self.create_network(state_size, action_size) 
        self.action_gradient = tf.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(LR).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())
    
    
    def train(self, states, action_grads):
        if self.mode == "LIDAR":
            self.sess.run(self.optimize, feed_dict={
                self.state_1: np.expand_dims(np.array(states[:,0:-1]),axis=2),
                self.state_2: np.array(states[:,-1]).reshape([-1,1]),
                self.action_gradient: action_grads
            })
        else:
            self.sess.run(self.optimize, feed_dict={
                self.state_1: np.array(states[:,0:2]),
                self.state_2: np.array(states[:,2:4]),
                self.action_gradient: action_grads
            })
        
    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)
        
    def create_network(self,state_size,action_dim):
        if self.mode =="LIDAR":
            print("Creating Actor NN - LIDAR mode")
            inp_1 = Input(shape=(state_size-1,1),name='state_1')
            inp_2 = Input(shape=(1,),name='state_2')
#            Branch 1
            conv_10 = Conv1D(filters=30,kernel_size=3,strides=1,padding="same",activation="relu")(inp_1)
            pool_11 = MaxPool1D()(conv_10)
            conv_12 = Conv1D(filters=30,kernel_size=3,strides=1,padding="same",activation="relu")(pool_11)
            pool_13 = MaxPool1D()(conv_12)
            flat_14 = Flatten()(pool_13)
#            Branch 2
            dense_20 = Dense(30,activation='relu')(inp_2)
            
#            Branch 1+2
            merge_30 = Concatenate()([flat_14,dense_20])
            dense_31 = Dense(60,activation='relu')(merge_30)
#            norm_32 =  BatchNormalization()(dense_31)
#            drop_33 = Dropout(0.5)(norm_32)
            dense_34 = Dense(30,activation='relu')(dense_31)
            Acceleration = Dense(1,activation='tanh')(dense_34)
            Steering = Dense(1,activation='tanh')(dense_34)
            V = Concatenate()([Steering,Acceleration])
            model = Model(inputs=[inp_1,inp_2],outputs=V)
        else:
            print("Creating Actor NN - GLOBAL mode")
            inp_1 = Input(shape=[state_size-2],name='state_1')
            inp_2 = Input(shape=[2],name='state_2')
            
            h00 = Dense(HIDDEN_1, activation='relu')(inp_1)
            h01 = Dense(HIDDEN_1, activation='relu')(h00)
            h02 = Dense(HIDDEN_2, activation='relu')(h01)
            
            h10 = Dense(HIDDEN_1,activation = 'relu')(inp_2)
            h11 = Dense(HIDDEN_2,activation = 'relu')(h10)
            h2 =  Concatenate()([h02,h11]) 
    #        TODO Add batchnormalisation
    #        TODO ADd dropout?
            h3 = Dense(HIDDEN_2, activation='relu')(h2)
            Acceleration = Dense(1,activation='tanh')(h3)
            Steering = Dense(1,activation='tanh')(h3)        
            V = Concatenate()([Steering,Acceleration])          
            model = Model(inputs=[inp_1,inp_2],outputs=V)
        return model, model.trainable_weights, inp_1, inp_2

