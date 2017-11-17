# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 19:47:52 2017

@author: Gergo
"""

#%%
import numpy as np

class Car:
    def __init__ (self,width,length,start_pos,speed,start_dir):
        self.width = width
        self.length = length
        self.pos=np.array(start_pos)
        self.speed=0
        self.dir=start_dir
        self.prev_acc = 0
        self.prev_steering = 0
    def reset (self,start_pos,speed,start_dir):
        self.pos=np.array(start_pos)
        self.speed=speed
        self.dir=start_dir
        self.prev_acc = 0
        self.prev_steering = 0
