#!/usr/bin/env python3

import numpy as np
import sklearn.feature_selection
import pickle

def ScrubArrays():

#STEPS
#0. Create a array of ones, length 110968486
#1. Import one subject's data
#2. In each position, if empty matrix & subject data are both nonzero, keep array=1
#3. Else make array position's value =0
#4. Extract the indices from array, cut subject arrays to match
    
    #SUBJECT ID LIST
    ones_array=np.ones(110968486)

    for subject in data:
        print('\nSubject ID: {}'.format(subject))
        scans=[]
        for i in glob.glob(data[subject]['dir'] + '/**/*.nii', recursive=True):
            scans.append(i)
            scans.sort()
        print('# of scans: {}'.format(len(scans)))
        vector_of_scans= []
        for i in scans:
            j= nib.load(i)
            scan= j.get_data()
            scan= np.reshape(scan, (1,-1))
            scan= np.concatenate(scan)
            vector_of_scans.extend(scan)
        vector_of_scans=np.array(vector_of_scans[0::])
        print('Vector length: {}'.format(len(vector_of_scans)))
        data[subject]['vector']=vector_of_scans
    
    

with open('/media/james/ext4data/current/projects/ramasubbu/data.pickle', 'rb') as f: data= pickle.load(f)

X=np.array([data[i]['vector'] for i in data])

data=[]

with open('/media/james/ext4data/current/projects/ramasubbu/X.pickle', 'wb') as d: pickle.dump(X, d, pickle.HIGHEST_PROTOCOL) 

#-- stop here and exit everything --

with open('/media/james/ext4data/current/projects/ramasubbu/X.pickle', 'rb') as f: X= pickle.load(f)

vt= sklearn.feature_selection.VarianceThreshold()

vt.fit(X)

nonzero_indices= vt.get_support(indices=True)

