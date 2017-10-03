# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 19:47:52 2017

@author: Gergo
"""

#%%
import numpy as np


class Car:
    def __init__ (self,width,height,gg_diag,start_pos,start_dir):
        self.width = width
        self.height = height
        self.pos=np.array(start_pos)
        self.speed=0
        self.dir=start_dir
        #        TODO gg_diag-ot beolvasni és elcachelni
    def reset (self,start_pos,start_dir,speed):
        self.pos=np.array(start_pos)
        self.speed=np.array(speed)
        self.dir=start_dir
