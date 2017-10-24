#!/usr/bin/env python3

import pickle, os


def agglomerate():
    
    
    bootstraps= os.listdir('/media/james/ext4data1/current/projects/ramasubbu/oct_24_results/bootstrap_results')
    
    results=[]
    
    for i in bootstraps:
        with open('/media/james/ext4data1/current/projects/ramasubbu/oct_24_results/bootstrap_results/'+i,'rb') as f: result= pickle.load(f)
        results.append(result)
        
    results.sort_values('p-value')    
        
    return     
