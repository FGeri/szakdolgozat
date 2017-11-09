# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:38:40 2017

@author: Gergo
"""

import random
import numpy as np 

class OU(object):

    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)