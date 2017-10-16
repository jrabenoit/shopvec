#!/usr/bin/env python3

import pickle
import pandas as pd
import numpy as np


def Stacker():
    bigtable=np.empty([52,91654728])
    counter=0
    for filename in os.listdir('/media/james/ext4data1/current/projects/ramasubbu/nonzerovectors'):
        with open('/media/james/ext4data1/current/projects/ramasubbu/vectors/'+filename, 'rb') as f: vector=pickle.load(f)
        bigtable[counter]= vector
        
        counter=counter+1

    return
