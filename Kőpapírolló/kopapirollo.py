# -*- coding: utf-8 -*-+
"""
Created on Wed Oct 18 10:41:44 2017
Piloting with Rock-Paper-Scissors in Keras
@author: Gergo
"""

#%%
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation,BatchNormalization
from keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np

def pick_action(states,strategy = None):
    if not strategy:
        actions = list(map(lambda x: 1,states))
    elif strategy =="LAST":
        actions = list(map(lambda x: ((x-(x//9**2)*(9**2)-(x//9**1)*(9**1))%3+1)%3,states))
    else:
        actions = list(map(lambda x: np.random.choice(3,1,p=[0.90,0.10,0]),states))
    return actions
    
    
np.random.seed(123)    
n = 2

states = ["kő-kő","kő-papír","kő-olló",
          "papír-kő","papír-papír","papír-olló",
          "olló-kő","olló-papír","olló-olló"]


actions = [0,1,2]
actions = np.reshape(actions,[-1,1])
states = np.array(states)

onehot_encoder = OneHotEncoder(sparse = False)
onehot_encoder.fit(np.reshape(np.array(range(9**n)),[-1,1]))

action_onehot_encoder = OneHotEncoder(sparse=False)
action_onehot_encoder.fit(actions)
#Creating the network
model = Sequential()
model.add(Dense(units = 81,input_dim = 9**n))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dense(units= 3))
model.add(Activation('softmax'))
model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                loss='categorical_crossentropy',metrics=['accuracy'])
# =============================================================================
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd,
#                loss='categorical_crossentropy',metrics=['accuracy'])
# =============================================================================


#Generate training data
x_train = np.random.choice(9**n,1000)
y_train = pick_action(x_train,"LAST")
x_test = np.random.choice(9**n,100)
y_test = pick_action(x_test,"LAST")

x_train = np.reshape(x_train,[-1,1])
y_train = np.reshape(y_train,[-1,1])
x_test = np.reshape(x_test,[-1,1])
y_test = np.reshape(y_test,[-1,1])

x_train = onehot_encoder.transform(x_train)
x_test = onehot_encoder.transform(x_test)
y_train = action_onehot_encoder.transform(y_train)
y_test = action_onehot_encoder.transform(y_test)

model.fit(x_train,y_train,epochs = 40,batch_size=32)
model.evaluate(x_test,y_test)


