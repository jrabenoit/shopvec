#!/usr/bin/env python3

from operator import itemgetter
import numpy as np
import pickle

#Gives average accuracy of each param set tested on inner loop data
def InnerAverages():
    with open('/media/james/ext4data1/current/projects/ramasubbu/svmscores.pickle','rb') as f: scores=pickle.load(f)

    avgs=np.add.reduceat(scores['test'], np.arange(0, 25, 5))*100/5
    
    print(avgs)
    
    return

def OuterAverages():
    with open('/media/james/ext4data1/current/projects/ramasubbu/gnbscores_outer.pickle','rb') as f: gnb=pickle.load(f)
    with open('/media/james/ext4data1/current/projects/ramasubbu/rfscores_outer.pickle','rb') as f: rf=pickle.load(f)
    
    gnb_outer=np.add.reduceat(gnb['inner test'], np.arange(0, 25, 5))*100/5
    rf_outer=np.add.reduceat(rf['inner test'], np.arange(0, 25, 5))*100/5
    
    print(gnb_inner)
    print(rf_inner)
    
    return 
