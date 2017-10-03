# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 19:33:26 2017

@author: Gergo
"""

#%%
import numpy as np
import Environment
import Car
from Frontend import *
from copy import deepcopy
#


def start():
    
    print('GOOOOOO!!!')
    
    



gui = GUI(handler=start_simulation)
gui.mainloop()



def start_simulation(GUI):
    
# =============================================================================
#     TODO Initialise our pandas database with max_simulation_data
# =============================================================================
    max_simulation_data = 1000
    env = Environment(GUI.track_path,np.array([[252,30],[252,80]]),
                np.array([[251,30],[251,80]]),[(0,0),(0,1)],np.array([255,255,255]),time_step=1)
    car = Car(GUI.width,GUI.height,GUI.gg_path,env.start_position,0)
    for i in np.arange(max_simulation_data):
# =============================================================================
#         TODO Randomly initialise (switchable) our starting 
# =============================================================================
        car.reset(env.start_position,0)
        over = False
                
        while over:
# =============================================================================
#           TODO  We get the our action here (acc, steer_angle)
# =============================================================================
             
            acc = 1
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
      
# =============================================================================
#         TODO Once we have the data, we split them into to sections.
#           Train data and validation data.We normalise them 
# =============================================================================
    
            
# =============================================================================
#   TODO We tran the network with the train data. (R only first then R+max(Q')).
#   Then at validation we stop when the error is the lowest
# =============================================================================


# =============================================================================
#   TODO With the trained network we test our algorythm
# =============================================================================















