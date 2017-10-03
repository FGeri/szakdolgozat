# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:58:34 2017

@author: Gergo
"""
import numpy as np
import math
import matplotlib
import keras
import scipy
from skimage import io
from PIL import ImageTk, Image

class Environment:
    track
    obstacles
    time_step
    start_line #np array [[startpoint x,y], [endpoint x,y]  
    finish_line #np array [[startpoint x,y], [endpoint x,y] 
    gg_diag
    color_of_track
    
    def __init__(self,path_of_track,start_line,
                 finish_line,obstacles,color_of_track,time_step=1):
        
        self.track=scipy.misc.imread(path_of_track)        
        self.start_line = np.array(start_line)
        self.finish_line = np.array(finish_line)
        
        self.start_position = np.array([round((start_line[0][0]+start_line[0][1])/2),
                                        round((start_line[1][0]+start_line[1][1])/2)])
        self.obstacles = set(obstacles)
        self.time_step = time_step
        self.color_of_track = color_of_track
    
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
        delta_dir = car.speed+acc / car.length * math.tan(steer_angle)
        delta_x =  car.speed+acc * math.sin(car.dir)
        delta_y =  car.speed+acc * math.cos(car.dir)
        
        car.speed = car.speed + acc
        car.dir = car.dir + delta_dir
        car.pos = car.pos + np.array([delta_x, delta_y])

# =============================================================================
#     This function checks if the car has collided with an obstacle or is out 
#     of the track
#     Returns true if collided
# =============================================================================
    def detect_collision(self,car):
        collision = False
        theta = car.dir
        rot_matrix = np.array([[-math.sin(theta),math.cos(theta)],[math.cos(theta),math.sin(theta)]]) 
        chassis = np.ones(car.height*2+1,car.width*2+1)
        indexes = np.where(chassis==1)
        indexes[0,:] -= car.height
        indexes[1,:] -= car.width
        size_of_mask = math.ceil(((car.height*2+1)**2+(car.width*2+1)**2)**0.5)
        mask = np.zeros([size_of_mask,size_of_mask])
        rotated_indexes = math.round(rot_matrix.dot(indexes))
        
#        rotated_indexes += size_of_mask
#        mask[rotated_indexes[0,:],rotated_indexes[1,:]]=1
        
        rotated_indexes[0,:] += car.pos[1]
        rotated_indexes[1,:] += car.pos[0]
        
        if np.any(rotated_indexes[0,:] < 0 or rotated_indexes[0,:]>=self.track.shape[0] or
                  rotated_indexes[1,:] < 0 or rotated_indexes[1,:]>=self.track.shape[1]):
            collision = True
        else:
            fields = self.track[rotated_indexes[0,:],rotated_indexes[1,:],:]
            
    #        TODO Check if works correctly
    #        np.all(np.all(a==np.array([[[1]],[[2]],[[3]]]),axis = 0))
            np.all(np.all(fields==np.array(self.color_of_track),axis = 2))
            
#            Checks if still on track
            if not np.all(np.all(fields==np.array(self.color_of_track),axis = 2)):
                collision = True
        
            rotated_indexes_trans=rotated_indexes.transpose()
            rotated_indexes_trans = list(map((lambda x : (x[1],x[0])),rotated_indexes_trans))
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
        if (prev_pos[0] < self.finish_line[0][0] and car.pos[0] >=self.finish_line[0][0] 
            and prev_pos[1] > self.finish_line[0][1] and prev_pos[1] < self.finish_line[1][1]):
            over = True
        elif (prev_pos[0] >= self.finish_line[0][0] and car.pos[0] < self.finish_line[0][0] 
            and prev_pos[1] > self.finish_line[0][1] and prev_pos[1] < self.finish_line[1][1]):
            over = True
            collision = True
        else:
            over = False
        
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
    