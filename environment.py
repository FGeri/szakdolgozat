# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:58:34 2017

@author: Gergo
"""


import numpy as np
import math
import matplotlib
from copy import copy,deepcopy
#import keras
import scipy
import math
import random
from skimage import io
from PIL import ImageTk, Image
from scipy.integrate import ode
from skimage.morphology import disk


# =============================================================================

#y = [x,y,v,dir]
#arg= [a,steering,l]
# =============================================================================

def derivatives(t, y, arg):
    dy = [0,0,0,0]
    dy[0] = y[2]*math.sin(y[3])
    dy[1] = -y[2]*math.cos(y[3])
    dy[2] = arg[0]
    dy[3] = -y[2]/arg[2]*math.tan(arg[1])
    return dy
    
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
        random.seed(123)
        self.track=scipy.misc.imread(path_of_track)        
        self.start_line = np.array(start_line)
        self.finish_line = np.array(finish_line)
        self.start_speed = 2
        self.start_dir = math.pi/2
#        self.start_dir = 0
        self.start_position = np.array([round((start_line[0,0]+start_line[1,0])/2),
                                        round((start_line[0,1]+start_line[1,1])/2)])
        self.obstacles = set(obstacles)
        self.time_step = copy(time_step)
        self.color_of_track = np.array(color_of_track)
        self.ode = ode(derivatives).set_integrator('dopri5')
        self.prev_lateral_dist = 0
        self.prev_longit_dist = 0
        self.dists = self.__get_dists()
        self.track_indexes = []
        for x in range(self.track.shape[1]):
            for y in range(self.track.shape[0]):
                if np.array_equal(self.track[y, x, :], self.color_of_track):
                    self.track_indexes.append([x, y])
        
    def reset (self,random_init = False):
        self.prev_lateral_dist = 0
        self.prev_longit_dist = 0
        if random_init:
            self.start_speed = np.random.uniform(-30,30)
            self.start_dir = np.random.uniform(0,math.pi*2)
            self.start_position = np.array(random.choice(self.track_indexes))
            self.get_reward(self.start_position)
        
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
        y0, t0 = [car.pos[0],car.pos[1],car.speed,car.dir], 0
        dt = self.time_step
        self.ode.set_initial_value(y0, t0).set_f_params([acc,steer_angle,car.length])
        self.ode.integrate(self.ode.t+dt)
        car.speed = self.ode.y[2]
        car.dir = (self.ode.y[3]) % (2*math.pi)
        car.pos = np.array([self.ode.y[0], self.ode.y[1]]).astype(float)

# =============================================================================
#     This function checks if the car has collided with an obstacle or is out 
#     of the track
#     Returns true if collided
# =============================================================================
    def __detect_collision(self,car):
        collision = False
        theta = car.dir
        rot_matrix = np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]],dtype=float) 
        chassis = np.ones([int(car.length),int(car.width)])
        indexes_tmp = np.where(chassis==1)
        indexes = np.array(indexes_tmp,dtype=float)
        indexes[0,:] -= math.floor((car.length-1)/2)
        indexes[1,:] -= math.floor((car.width-1)/2)
        indexes=indexes[::-1,:]
        rotated_indexes = rot_matrix.dot(indexes)
        rotated_indexes[0,:] += car.pos[0]
        rotated_indexes[1,:] += car.pos[1]
        rotated_indexes = np.round(rotated_indexes)
        rotated_indexes = rotated_indexes.astype(int)
        if (np.any(rotated_indexes[0,:] < 0) or np.any(rotated_indexes[0,:]>=self.track.shape[1]) or
                  np.any(rotated_indexes[1,:] < 0) or np.any(rotated_indexes[1,:]>=self.track.shape[0])):
            collision = True
        else:   
            fields = self.track[rotated_indexes[1,:],rotated_indexes[0,:],:]            
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
        collision = self.__detect_collision(car)
        if (prev_pos[0] < self.start_line[0,0] and car.pos[0] >=self.start_line[0,0] 
            and prev_pos[1] > self.start_line[0,1] and prev_pos[1] < self.finish_line[1,1]):
            over = True
        elif (prev_pos[0] >= self.start_line[0,0] and car.pos[0] < self.start_line[0,0] 
            and prev_pos[1] > self.start_line[0,1] and prev_pos[1] < self.start_line[1,1]):
            over = True
            collision = True
        else:
            over = collision
        
        return over, not collision

# =============================================================================
#     Gets sensor data
# =============================================================================
    def get_sensor_data(self, pos, ref_dir, resolution, max_r):
        track = deepcopy(self.track)
        for obstacle in self.obstacles:
            track[obstacle[1],obstacle[1],:] = 0
        sensor_data = np.array([[0 for i in range(resolution+1)],
                               [-math.pi/2+i*math.pi/resolution for i in range(resolution+1)],
                               [0 for i in range(resolution+1)]])
        sensor_data[1,:]=sensor_data[1,:]+ref_dir
        points = np.array([np.sin(sensor_data[1,:]),-np.cos(sensor_data[1,:])])
        r = 0
        ready = pos[0] < 0 or pos[1] < 0 or \
                pos[0] >= track.shape[1] or pos[1] >= track.shape[0] or \
                np.any(track[np.round(pos[1]).astype(int),np.round(pos[0]).astype(int),:]!=np.array([255,255,255]),axis = -1)
        while  not ready :
                r = r+1
                points_to_check = np.array(points*r+pos.reshape(-1,1))
                points_to_check = np.round(points_to_check).astype(int)
                sensor_data[2,:] =  (sensor_data[2,:]) + \
                                    (points_to_check[0,:] < 0) + (points_to_check[1,:] < 0) + \
                                    (points_to_check[0,:] >= track.shape[1]) + (points_to_check[1,:] >= track.shape[0]) + \
                                    (np.any(track[list(map(lambda x : min(x,track.shape[0]-1),points_to_check[1,:])),
                                                  list(map(lambda x : min(x,track.shape[1]-1),points_to_check[0,:])),
                                                  :]!=np.array([255,255,255]),axis = -1))
                sensor_data[0,:] = [sensor_data[0,i]+1 \
                                    if sensor_data[2,i] == 0 \
                                    else sensor_data[0,i] \
                                    for i in range(len(sensor_data[0,:]))]
                ready = np.all(sensor_data[2,:]) or r == max_r
        return sensor_data[0,:]
# =============================================================================
#     Computes the reward based on the position of the car
# =============================================================================
    def get_reward(self, pos_original):
        
            pos = np.array(pos_original, dtype='int32')
            tmp = [0]
            r = 0
            inner_flag = True
            outer_flag = True
            while inner_flag or outer_flag:
                r = r+1
                dx =[0,0]
                dy =[0,0]
                if pos[0]-r < 0:
                    dx[0] = -(pos[0]-r)
                if pos[0]+r+1>self.track.shape[1]:
                    dx[1] = pos[0]+r+1-self.track.shape[1]
                if pos[1]-r < 0:
                    dy[0] = -(pos[1]-r)
                if pos[1]+r+1>self.track.shape[0]:
                    dy[1] = pos[1]+r+1-self.track.shape[0]
                y = [max(pos[1]-r,0),min(pos[1]+r+1,self.track.shape[0])] 
                x = [max(pos[0]-r,0),min(pos[0]+r+1,self.track.shape[1])]
                tmp = self.track[y[0]:y[1], x[0]:x[1],:]
                mask = disk(r)
                mask = np.expand_dims(mask,2)
                mask = np.repeat(mask, 3,axis = 2)
                tmp = mask[dy[0]:mask.shape[0]-dy[1],dx[0]:mask.shape[1]-dx[1],:] * tmp
                next_inner_flag = not np.any(np.all(tmp==np.array([0,0,255]),axis = 2))
                next_outer_flag = not np.any(np.all(tmp==np.array([195,195,195]),axis = 2))
                if inner_flag and not next_inner_flag:
                    section = deepcopy(tmp)
                    inner_dist=deepcopy(r)
                if outer_flag and not next_outer_flag:
                    outer_dist=deepcopy(r)
                inner_flag=deepcopy(next_inner_flag)
                outer_flag=deepcopy(next_outer_flag)
            r=inner_dist
            section = np.all(section==np.array([0,0,255]),axis = 2)    
            section = section.astype(int)
            indexes = [p[0] for p in np.nonzero(section)]
            offset = [indexes[1]-r+dx[0], indexes[0]-r+dy[0]]
            pos = np.array(pos + offset)
            track_width= inner_dist+outer_dist
            longit_dist = self.dists[tuple(pos)]
            lateral_dist = abs(inner_dist/track_width-0.5)
            reward = (longit_dist - self.prev_longit_dist)-(lateral_dist-self.prev_lateral_dist)*10
            self.prev_longit_dist = longit_dist
            self.prev_lateral_dist = lateral_dist
            return reward
        
    def __get_dists(self):
        """
        :return: a dictionary consisting (inner track point, distance) pairs
        """
        dist_dict = {}        
#        start_point = np.array([252,72])
        start_point = np.array(self.start_position).astype(int)
        tmp = [0]
        r = 0
        flag = True
        while flag:
            r = r+1
            tmp = self.track[start_point[1]-r:start_point[1]+r+1, start_point[0]-r:start_point[0]+r+1,:]
            mask = disk(r)
            mask = np.expand_dims(mask,2)
            mask = np.repeat(mask, 3,axis = 2)
            tmp = mask * tmp
            flag = not np.any(np.all(tmp==np.array([0,0,255]),axis = 2))
        tmp = np.all(tmp==np.array([0,0,255]),axis = 2)
        indexes = [p[0] for p in np.nonzero(tmp)]
        offset = [indexes[1]-r, indexes[0]-r]
        start_point = np.array(start_point+offset)
        dist_dict[tuple(start_point)] = 0
        dist = 0
        RIGHT, UP, LEFT, DOWN = [1, 0], [0, -1], [-1, 0], [0, 1]  # [x, y], tehát a mátrix indexelésekor fordítva, de a pozícióhoz hozzáadható azonnal
        dirs = [RIGHT, UP, LEFT, DOWN]
        direction_idx = 0
        point = start_point
        dist_dict[tuple(point)] = 0
        while True:
            dist += 1
            left_turn = dirs[(direction_idx+1) % 4]
            right_turn = dirs[(direction_idx-1) % 4]
            if np.all(self.track[point[1] + left_turn[1], point[0] + left_turn[0],:] == np.array([0,0,255])):
                direction_idx = (direction_idx + 1) % 4
                point = point + left_turn
            elif np.all(self.track[point[1] + dirs[direction_idx][1], point[0] + dirs[direction_idx][0],:] == np.array([0,0,255])):
                point = point + dirs[direction_idx]
            else:
                direction_idx = (direction_idx - 1) % 4
                point = point + right_turn
            if np.array_equal(point, start_point):
                break
            dist_dict[tuple(point)] = dist
        return dist_dict