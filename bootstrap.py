#!/usr/bin/env python3 

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pprint, itertools, pickle, random, statistics

#Because we are sampling with replacement, we don't need to worry about the program picking all subjects each time... some may be picked more than once, and the total number of samples will be equal to the number of subjects.

def Bill():
    with open('/media/james/ext4data1/current/projects/ramasubbu/inner_test_results.pickle','rb') as f: inner_test=pickle.load(f)
    
    g=inner_test.groupby('subjects')    
    
    #Per subject accuracy
    acc= (g['scores'].sum()/g['attempts'].sum())*100
    n= len(acc)
    runs= 10000

    distribution= []
    for i in range(runs):        
        sample= np.random.choice(acc, n,replace=True) 
        sample_mean= sum(sample)/len(sample)
        distribution.append(sample_mean)
    
    print('{} runs of {} samples'.format(len(distribution), n))
    print('distribution mean: {}%'.format(sum(distribution)/len(distribution)))
    print('p-value: {}'.format(sum(i<=50 for i in distribution)/runs))
    
    binner=np.digitize(distribution, np.array(range(0,101)))
    plt.plot([50,50],[0,list(binner).count(statistics.mode(binner))],'-r',lw=2)
    plt.hist(distribution, bins=list(range(0,101)))
    plt.show()
    
    return
