#!/usr/bin/env python3

import numpy as np
from sklearn import preprocessing

def ZNormalize(masked_data):
    '''Will this work with sparse data? See 4.3.1.2
    Also do we fit the training set then transform the training & test sets?'''
    stdscaler = StandardScaler(copy=True, with_mean=True, with_std=True)
    
    
    
    return znorm_data
