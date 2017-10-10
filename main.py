# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 19:33:26 2017

@author: Gergo
"""

#%%
%load_ext autoreload
%autoreload 2
import time
import numpy as np
import math
import matplotlib
import matplotlib.pyplot  as plt
from environment import Environment
from car import Car
import pandas as pd
#import keras
from frontend import *
from copy import deepcopy

def start_simulation(GUI):
    
    max_simulation_laps = 10
    env = Environment(GUI.track_path.get(),np.array([[252,30],[252,80]]),
                np.array([[251,30],[251,80]]),[(0,0),(0,1)],np.array([255,255,255]),time_step=1)
    car = Car(GUI.width.get(),GUI.length.get(),GUI.gg_path.get(),env.start_position,env.start_speed,env.start_dir)
    data=[]
    theta = car.dir
    rot_matrix = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]],dtype=float)
    # =============================================================================
#             Draw the track and car
# =============================================================================
    chassis = np.array([[0,car.width,car.width,0],[0,0,car.length,car.length]],dtype=float)
    chassis = rot_matrix.dot(chassis)
    chassis[0,:] = chassis[0,:] + car.pos[0] - car.width/2 
    chassis[1,:] = chassis[1,:] + car.pos[1] - car.length/2
    chassis= np.round(chassis)
    chassis = chassis.astype(int)
    GUI.track_figure_handle.clear()
    track = GUI.track_figure_handle.add_subplot(111)
    plt.imshow(GUI.track_img,aspect='auto')
    plt.plot([env.start_line[0,0],env.start_line[1,0]], [env.start_line[0,1],
             env.start_line[1,1]], color='g', linestyle='-', linewidth=1)
    plt.plot([chassis[0,0],chassis[0,1]], [chassis[1,0],chassis[1,1]],
             color='k', linestyle='-', linewidth=2)
    plt.plot([chassis[0,1],chassis[0,2]], [chassis[1,1],chassis[1,2]],
             color='k', linestyle='-', linewidth=2)
    plt.plot([chassis[0,2],chassis[0,3]], [chassis[1,2],chassis[1,3]],
             color='k', linestyle='-', linewidth=2)
    plt.plot([chassis[0,3],chassis[0,0]], [chassis[1,3],chassis[1,0]],
             color='k', linestyle='-', linewidth=2)
    GUI.draw_track_callback()
    GUI.update_idletasks()
    GUI.update()
    for i in np.arange(max_simulation_laps):
# =============================================================================
#         TODO Randomly initialise (switchable) our starting 
# =============================================================================
        env.reset(False)
        car.reset(env.start_position,env.start_speed,env.start_dir)
        over = False
#        tmp=1
        while not over:
# =============================================================================
#           TODO  We get the our action here (acc, steer_angle)
# =============================================================================
#            acc = np.random.uniform(-30,30)
#            steer_angle = np.random.uniform(-math.pi/2,math.pi/2)
            acc = 0
            steer_angle = math.pi/(4*4)
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
            
            car_color = 'r' if over and not result else 'k'
# =============================================================================
#             Draw the track and car
# =============================================================================
            theta = car.dir
            rot_matrix = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]],dtype=float)
            chassis = np.array([[0,car.width,car.width,0],[0,0,car.length,car.length]])
            chassis[0,:] -= car.width/2
            chassis[1,:] -= car.length/2
            chassis = rot_matrix.dot(chassis)
            chassis[0,:] = chassis[0,:] + car.pos[0] 
            chassis[1,:] = chassis[1,:] + car.pos[1]
            chassis= np.round(chassis)
            chassis = chassis.astype(int)
            GUI.track_figure_handle.clear()
            track = GUI.track_figure_handle.add_subplot(111)
            plt.imshow(GUI.track_img,aspect='auto')
            plt.plot([env.start_line[0,0],env.start_line[1,0]], [env.start_line[0,1],
                      env.start_line[1,1]], color='g', linestyle='-', linewidth=1)
            plt.plot([chassis[0,0],chassis[0,1]], [chassis[1,0],chassis[1,1]],
                     color=car_color, linestyle='-', linewidth=2)
            plt.plot([chassis[0,1],chassis[0,2]], [chassis[1,1],chassis[1,2]],
                     color=car_color, linestyle='-', linewidth=2)
            plt.plot([chassis[0,2],chassis[0,3]], [chassis[1,2],chassis[1,3]],
                     color=car_color, linestyle='-', linewidth=2)
            plt.plot([chassis[0,3],chassis[0,0]], [chassis[1,3],chassis[1,0]],
                     color=car_color, linestyle='-', linewidth=2)
            GUI.draw_track_callback()
            GUI.update_idletasks()
            GUI.update()

            
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
# =============================================================================
# while True:
#     gui.update_idletasks()
#     gui.update()
#     time.sleep(0.1)
# =============================================================================
gui.mainloop()





