# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 15:58:34 2017

@author: Gergo
"""
import matplotlib
from skimage import io

class Environment:
    track
    obstacles
    time_step
    start_line
    finish_line
    
    
    def __init__(self,path_of_track):
        self.track=io.imread(path_of_track);
    
    def step(self,car):
        pass
    
    def detectCollision(self,car):
        pass
    
    def isOver(self,car):
        pass
    
    def getReward(self):
        pass
    