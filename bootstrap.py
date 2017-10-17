#!/usr/bin/env python3 

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pprint, itertools, pickle, random, statistics

#Because we are sampling with replacement, we don't need to worry about the program picking all subjects each time... some may be picked more than once, and the total number of samples will be equal to the number of subjects.

def Bill(group, run):
    with open('/media/james/ext4data1/current/projects/ramasubbu/outer_test_results_group_'+str(group)+'_run_'+str(run)+'.pickle','rb') as f: inner_test=pickle.load(f)
    
    #Per subject accuracy
    g=inner_test.groupby('subjects')    
    acc= (g['scores'].sum()/g['attempts'].sum())*100
    n= len(acc)
    runs= 10000

    distribution= []
    for i in range(runs):        
        sample= np.random.choice(acc, n,replace=True) 
        sample_mean= sum(sample)/len(sample)
        distribution.append(sample_mean)
    
    dist_mean= sum(distribution)/len(distribution)
    p_value= sum(i<=50 for i in distribution)/runs
    
    print('{} runs, {} samples per run'.format(len(distribution), n))
    print('distribution mean: {}%'.format(dist_mean))
    print('p-value: {}'.format(p_value))
    
    bootstrap_results= {'samples per run': n, 
                        'runs': 10000, 
                        'distribution mean': dist_mean, 
                        'p-value': p_value
                        }
    
    #binner=np.digitize(distribution, np.array(range(0,101)))
    #plt.plot([50,50],[0,list(binner).count(statistics.mode(binner))],'-r',lw=2)
    #plt.hist(distribution, bins=list(range(0,101)))
    #plt.show()
    
    with open('/media/james/ext4data1/current/projects/ramasubbu/bootstrap_results_group_'+str(group)+'_run_'+str(run)+'.pickle', 'wb') as d: pickle.dump(train_results, d, pickle.HIGHEST_PROTOCOL) 
    
    return
