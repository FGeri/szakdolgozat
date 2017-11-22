# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 10:50:56 2017

@author: Gergo
"""
import numpy as np
import pandas as pd
from collections import deque
import random
from copy import deepcopy

def softmax(a):
    x = np.asarray(a,dtype=float)
    exps = np.exp(x)
    return  exps/exps.sum()

class Buffer:
    def __init__(self,buffer_size,enable_per):
        self.buffer = pd.DataFrame([[np.array([]),np.array([]),0.,np.array([]),False,"",0.]],columns=['s', 'a', 'r', "s'", 'over','id','p'])
        self.buffer = self.buffer.drop(self.buffer.index[0])
        self.size = buffer_size
        self.per = enable_per
        self.num_items = 0
        self.e = 0.01
        self.alfa = 0.0
        
    def size(self):
        return self.size
    
    def _getPriority(self, error):
        return (error + self.e) ** self.alfa

    def update_priorities(self,indeces,errors):
        priorities = self._getPriority(np.abs(errors))
        self.buffer.loc[indeces,'p'] = priorities

    def get_batch(self,batch_size):
        if self.num_items < batch_size:
            batch = self.buffer
        else:
            if self.enable_per:
                p = self.buffer.loc[:,'p']
                p = p / np.sum(p)
                batch = self.buffer.sample(batch_size,replace=False,weights = p)
            else:
                batch = self.buffer.sample(batch_size,replace=False)
        return batch.values[:],batch.index.values
        
    def add_item(self,item):
        item.loc[:,'p']=self._getPriority(abs(item.loc[:,'r']))
        if self.size > self.num_items:
            self.buffer=pd.concat([self.buffer,item],ignore_index=True)
        else:
#            p = self.buffer.loc[:,'r'].values
#            p = p * -1
#            p = softmax(p)
#            index = np.random.choice(range(len(p)),1,p=p,replace=False)
#            self.buffer = self.buffer.drop(self.buffer.index[index])
#            self.buffer = self.buffer.drop(self.buffer.index[-1])
#            self.buffer = pd.concat([self.buffer,item],ignore_index=True)
            pass
#        self.buffer.drop_duplicates('id',keep='first' ,inplace = True)
#        self.buffer.reset_index(inplace = True,drop = True)
        self.num_items = self.buffer.shape[0]
            

    def reset(self):
         self.buffer = self.buffer.drop(self.buffer.index[:])
         self.num_items = 0

