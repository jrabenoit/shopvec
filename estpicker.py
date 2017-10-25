#!/usr/bin/env python3 

import pickle
import numpy as np

def Best(group, run):
    with open('/media/james/ext4data1/current/projects/ramasubbu/inner_test_results_group_'+str(group)+'_run_'+str(run)+'.pickle', 'rb') as f: itr=pickle.load(f)
    
    g=itr.groupby('estimator')    
     
    estacc= (g['scores'].sum()/g['attempts'].sum())

    itr['diff']=[itr['scores'][i]-estacc[itr['estimator'][i]] for i in itr.index]    
    itr['sqdiff']=[itr['diff'][i]**2 for i in itr.index]
    estvar= g['sqdiff'].sum()

    n=len(itr['subjects'].unique())
        
    #>>>add code here to select based on best variance within best accuracy<<<
    maxacc=np.argmax(estacc)
    minvar= np.argmin(estvar)
    
    bestest= maxacc
    
    return bestest
