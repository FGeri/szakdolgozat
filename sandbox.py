# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 22:34:03 2017

@author: Gergo
"""

#%%

import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

device_name = sys.argv[1]  # Choose device from cmd line. Options: gpu or cpu
shape = (int(sys.argv[2]), int(sys.argv[2]))
if device_name == "gpu":
    device_name = "/gpu:0"
else:
    device_name = "/cpu:0"

with tf.device(device_name):
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)


startTime = datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)
        print(result)

# It can be hard to see the results on the terminal with lots of output -- add some newlines to improve readability.
print("\n" * 5)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", datetime.now() - startTime)

print("\n" * 5)
#%%
import math
from scipy.integrate import ode
# =============================================================================
# x
# y
# vx
# vy
# fi
# =============================================================================
y0, t0 = [0,1,1,0,0], 0
def f(t, y, arg1):
    dy = [0,0,0,0,0]
    dy[0] = y[2]
    dy[1] = y[3]
    dy[2] = -(y[2]**2+y[3]**2)*math.sin(y[4]) 
    dy[3] = -(y[2]**2+y[3]**2)*math.cos(y[4])  
    dy[4] = ((y[2]**2+y[3]**2)**0.5)
    return dy

r = ode(f).set_integrator('dopri5')
r.set_initial_value(y0, t0).set_f_params(2.0)
t1 = 2*math.pi
dt = math.pi/2
step = 0
while r.successful() and r.t < t1:
    step +=1
    r.integrate(r.t+dt)
    print('STEP: %s',step)
    print(r.t)
    print(r.y)
r.set_initial_value(y0, t0).set_f_params(2.0)
t1 = 2*math.pi
dt = math.pi/2
step = 0

#%%

x = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3],[4,4,4,4]])
x = np.expand_dims(x,2 )
y = np.repeat(x, 3,axis = 2)
#not np.all(
tmp = np.all(y==np.array([4,4,4]),axis = 2)
tmp=tmp.astype(int)
tmp[tmp==1]=10
print(np.nonzero(tmp))
indexes = np.array([p for p in np.nonzero(tmp)])
indexes
#%%
theta = math.pi/32
rot_matrix = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]],dtype=float)
chassis = np.array([[ 0,8,8,0],[0,0,16,16]],dtype=float)

chassis[0,:] -= 8/2
chassis[1,:] -= 16/2
chassis = rot_matrix.dot(chassis)
chassis[0,:] = chassis[0,:] + 60 - 8/2 
chassis[1,:] = chassis[1,:] + 60 - 16/2
print(chassis)
chassis= np.round(chassis)
chassis = chassis.astype(int)
print(chassis)
#%%

from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
# define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
print(values)
# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
print(integer_encoded)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)

#%%
from collections import deque
import random
buffer = deque()
buffer.append(["Kutya","Cica"])
print(random.sample(buffer,1))
#%%
import numpy as np

a = np.array([2,3])
print(max(list(a[:]),0))
