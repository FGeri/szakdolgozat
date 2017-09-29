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
    start_line
    finish_line
    start_position
    gg_diag
    color_of_track
    
    def __init__(self,path_of_track,start_line,
                 finish_line,start_position,obstacles,color_of_track,time_step=1):
        self.track=io.imread(path_of_track)
        self.start_line = np.array(start_line)
        self.finish_line = np.array(finish_line)
        self.start_position = np.array(start_position)
        self.obstacles = np.array(obstacles)
        self.time_step = time_step
        self.color_of_track = color_of_track
    
    def load_gg(self , path):
        self.gg_diag = scipy.misc.imread(path)
    
    def load_track(self , path):
        self.track = scipy.misc.imread(path)
    
    def load_nn(self , path):
        pass
    
    
    def step(self,car,acc,steer_angle):
        
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
    def detectCollision(self,car):
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
            
            if not np.all(np.all(fields==np.array(self.color_of_track),axis = 2)):
                collision = True
        
#        TODO Check collision with obstacles
        return collision
    
# =============================================================================
#     Checks if a game is over by runnin out of the track or colliding 
#     with an obstacles or reaching the finish
#     Returns: 
#       isOver:True if the lap is over
#       collision: False if the car reached the finish
# =============================================================================
    def isOver(self,car):
        pass
    
    
# =============================================================================
#     
# =============================================================================
    def getReward(self):
        pass
    