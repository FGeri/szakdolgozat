# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:50:56 2017

@author: Gergo
"""

from collections import deque
import random
from copy import deepcopy

class Buffer:
    def __init__(self,buffer_size):
        self.buffer = deque()
        self.size = buffer_size
        self.num_items = 0
        
    def size(self):
        return self.size

    def get_batch(self,batch_size):
        if self.num_items < batch_size:
            batch = random.sample(self.buffer, self.num_items)
        else:
            batch = random.sample(self.buffer, batch_size)
        return batch
        
    def add_item(self,item):
        if self.size > self.num_items:
            self.buffer.append(deepcopy(item))
            self.num_items += 1
        else:
            self.buffer.popleft()
            self.buffer.append(deepcopy(item))
            
            
    def reset(self):
        self.buffer.clear()
        self.num_items = 0