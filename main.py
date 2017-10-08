# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 19:33:26 2017

@author: Gergo
"""

#%%

import numpy as np
import math
from environment import Environment
from car import Car
import pandas as pd
#import keras
from frontend import *
from copy import deepcopy

def start_simulation(GUI):
    max_simulation_laps = 5
    env = Environment(GUI.track_path.get(),np.array([[252,30],[252,80]]),
                np.array([[251,30],[251,80]]),[(0,0),(0,1)],np.array([255,255,255]),time_step=1)
    car = Car(GUI.width.get(),GUI.length.get(),GUI.gg_path.get(),env.start_position,env.start_speed,env.start_dir)
    data=[]
    for i in np.arange(max_simulation_laps):
# =============================================================================
#         TODO Randomly initialise (switchable) our starting 
# =============================================================================
        env.reset()
        car.reset(env.start_position,env.start_speed,env.start_dir)
        over = False
        tmp=1
        while not over and tmp<30:
# =============================================================================
#           TODO  We get the our action here (acc, steer_angle)
# =============================================================================
            tmp+=1
            acc = 0
            steer_angle = 0
            prev_pos = deepcopy(car.pos)
            prev_speed = deepcopy(car.speed)
            prev_dir = deepcopy(car.dir)
            env.step(car,acc,steer_angle)
            over,result = env.is_over(car,prev_pos)
            
# =============================================================================
#             TODO Store our data in the pandas database.(s,a,r,s')
#            (prev_pos,prev_dir,prev_speed),(acc,steer_angle),r,(car.pos,car.dir,car.speed)
#            OR local info?
# =============================================================================

            state = np.concatenate([prev_pos[:],np.array([prev_speed,prev_dir])],axis=0)
            action = np.array([acc, steer_angle])
# =============================================================================
#             Observing reward
# =============================================================================
            if over and result:
                r = 50
            elif over and not result:
                r = -10
            elif not over:
#                print(car.pos)
                r = env.get_reward(car.pos)
            r = np.expand_dims(r,axis = 0)
            next_state = np.concatenate([car.pos[:],np.array([car.speed,car.dir])],axis=0)
            experience_raw = np.concatenate([state[:],action[:],r,next_state[:]],axis=0)
            experience=pd.DataFrame(np.expand_dims(experience_raw,axis = 0),
                columns = ["s1","s2","s3","s4","a1","a2","r","s'1","s'2","s'3","s'4"])
            data.append(experience)            
# =============================================================================
#         TODO Once we have the data, we split them into two sections.
#           Train data and validation data.We normalise them 
# =============================================================================
    data = pd.concat(data,ignore_index=True)
    print(data)        
# =============================================================================
#   TODO We tran the network with the train data. (R only first then R+max(Q')).
#   Then at validation we stop when the error is the lowest
# =============================================================================


# =============================================================================
#   TODO With the trained network we test our algorythm
# =============================================================================





# =============================================================================
# Main program
# =============================================================================

gui = GUI(handler=start_simulation)
gui.mainloop()







