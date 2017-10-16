#!/usr/bin/env python3 

import pickle
import numpy as np

def Best(group, run):
    with open('/media/james/ext4data1/current/projects/ramasubbu/inner_test_results_group_'+str(group)+'_run_'+str(run)+'.pickle', 'rb') as f: itr=pickle.load(f)
    
    g=itr.groupby('estimator')    
    
    acc= (g['scores'].sum()/g['attempts'].sum())*100
    print(acc)
    
    bestest=np.argmax(acc)
    
    return bestest
