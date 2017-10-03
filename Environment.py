# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:58:34 2017

@author: Gergo
"""


import numpy as np
import math
import matplotlib
from copy import copy
#import keras
import scipy
from skimage import io
from PIL import ImageTk, Image

class Environment:
# =============================================================================
#     track
#     obstacles
#     time_step
#     start_line #np array [[startpoint x,y], [endpoint x,y]  
#     finish_line #np array [[startpoint x,y], [endpoint x,y] 
#     gg_diag
#     color_of_track
# =============================================================================
    
    def __init__(self,path_of_track,start_line,
                 finish_line,obstacles,color_of_track,time_step=1):
        
        self.track=scipy.misc.imread(path_of_track)        
        self.start_line = np.array(start_line)
        self.finish_line = np.array(finish_line)
        
        self.start_position = np.array([round((start_line[0,0]+start_line[1,0])/2),
                                        round((start_line[0,1]+start_line[1,1])/2)])
        self.obstacles = set(obstacles)
        self.time_step = copy(time_step)
        self.color_of_track = np.array(color_of_track)
    
    def load_gg(self , path):
        self.gg_diag = scipy.misc.imread(path)
    
    def load_track(self , path):
        self.track = scipy.misc.imread(path)
    
    def load_nn(self , path):
        pass
    
    
# =============================================================================
#     Initialises the obstacles,stores them in a list with tupple elements
#     Options: static/dynamic
#              number of obstacles
#              Size?
#              
# =============================================================================
    def init_obstacles(self,options):
        pass
    
    
    def step(self,car,acc,steer_angle):
# =============================================================================
#       TODO Review the algorythm
# =============================================================================
        delta_dir = (car.speed+acc) / car.length * math.tan(steer_angle)
        delta_x =  (car.speed+acc) * math.sin(car.dir)
        delta_y =  (car.speed+acc) * (-math.cos(car.dir))
        car.speed = car.speed + acc
        car.dir = car.dir + delta_dir
        car.pos = car.pos + np.array([delta_x, delta_y]).astype(int)

# =============================================================================
#     This function checks if the car has collided with an obstacle or is out 
#     of the track
#     Returns true if collided
# =============================================================================
    def detect_collision(self,car):
        collision = False
        theta = car.dir
        rot_matrix = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]]) 
        chassis = np.ones([round(car.length*2+1),round(car.width*2+1)])
        indexes_tmp = np.where(chassis==1)
        indexes = np.array(indexes_tmp)
        indexes[0,:] -= int(car.length)
        indexes[1,:] -= int(car.width)
        size_of_mask = math.ceil(((car.length*2+1)**2+(car.width*2+1)**2)**0.5)
        mask = np.zeros([size_of_mask,size_of_mask])
        indexes=indexes[::-1,:]
        rotated_indexes = np.round(rot_matrix.dot(indexes))
        
#        rotated_indexes += size_of_mask.
#        mask[rotated_indexes[0,:],rotated_indexes[1,:]]=1
        
        rotated_indexes[0,:] += int(car.pos[0])
        rotated_indexes[1,:] += int(car.pos[1])
        if (np.any(rotated_indexes[0,:] < 0) or np.any(rotated_indexes[0,:]>=self.track.shape[1]) or
                  np.any(rotated_indexes[1,:] < 0) or np.any(rotated_indexes[1,:]>=self.track.shape[0])):
            collision = True
        else:
            rotated_indexes=rotated_indexes.astype(int)
            
            fields = self.track[rotated_indexes[1,:],rotated_indexes[0,:],:]

    #        TODO Check if works correctly
    #        np.all(np.all(a==np.array([[[1]],[[2]],[[3]]]),axis = 0))
#            np.all(np.all(fields==np.array(self.color_of_track),axis = 1))
            
#            Checks if still on track
            if not np.all(np.all(fields==np.array(self.color_of_track),axis = 1)):
                collision = True
        
            rotated_indexes_trans=rotated_indexes.transpose()
            rotated_indexes_trans = list(map((lambda x : (x[0],x[1])),rotated_indexes_trans))
            rotated_indexes_set = set(rotated_indexes_trans)
            if not self.obstacles.isdisjoint(rotated_indexes_set):
                collision = True
        return collision
    
# =============================================================================
#     Checks if a game is over by runnin out of the track or colliding 
#     with an obstacles or reaching the finish
#     Returns: 
#       over:True if the lap is over
#       collision: False if the car reached the finish
# =============================================================================
    def is_over(self,car,prev_pos):
        collision = self.detect_collision(car)
        if (prev_pos[0] < self.finish_line[0,0] and car.pos[0] >=self.finish_line[0,0] 
            and prev_pos[1] > self.finish_line[0,1] and prev_pos[1] < self.finish_line[1,1]):
            over = True
        elif (prev_pos[0] > self.finish_line[0,0] and car.pos[0] <= self.finish_line[0,0] 
            and prev_pos[1] > self.finish_line[0,1] and prev_pos[1] < self.finish_line[1,1]):
            over = True
            collision = True
        else:
            over = collision
        
        return over, collision
# =============================================================================
#     Updates the picture on the simulator page
# =============================================================================
    def draw_track(self):
        pass


# =============================================================================
#     Computes the reward based on the position of the car
# =============================================================================
    def get_reward(self):
        pass
    