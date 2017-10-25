#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle, os, statistics

def agglomerate():
    
    
    bootstraps= os.listdir('/media/james/ext4data1/current/projects/ramasubbu/oct_24_results/bootstrap_results')
    
    results=[]
    
    for i in bootstraps:
        with open('/media/james/ext4data1/current/projects/ramasubbu/oct_24_results/bootstrap_results/'+i,'rb') as f: result= pickle.load(f)
        results.append(result)
        
    results=pd.DataFrame(results)    
    results=results.sort_values('p-value')    

    res=results.loc[:,'p-value']
    res.plot()
    plt.show()
        
    return     
