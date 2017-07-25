#!/usr/bin/env python3

import numpy as np
import sklearn.feature_selection
import pickle, os, glob

def ScrubArrays():

#STEPS
#0. Create a array of ones, length 110968486
#1. Import one subject's data
#2. In each position, if empty matrix & subject data are both nonzero, keep array=1
#3. Else make array position's value =0
#4. Extract the indices from array, cut subject arrays to match

#beginning code
'''    
    #SUBJECT ID LIST
    oa=np.ones(110968486, dtype=np.float32)
    for filename in os.listdir('/media/james/ext4data1/current/projects/ramasubbu/vectors'):
        print('\nSubject ID: {}'.format(filename))
        #sort scan directory
        scans=[]
        for i in glob.glob(data[filename]['dir'] + '/**/*.nii', recursive=True):
            scans.append(i)
            scans.sort()
        print('# of scans: {}'.format(len(scans)))
        vecs= []
        #import a single subject's scans, create vector
        for i in scans:
            j= nib.load(i)
            scan= j.get_data()
            scan= np.reshape(scan, (1,-1))
            scan= np.concatenate(scan)
            vecs.extend(scan)
        vecs=np.array(vector_of_scans[0::])
        print('Vector length: {}'.format(len(vecs)))
        #compare master array to subject's vector
        for i,j in zip(ones_array, vector_of_scans):
            if ones_array[i]>0.0 or vector_of_scans[j]==0:
                ones_array[i]=0
        print('Remaining features: {}'.format(sum(ones_array)))
                
        data[subject]['vector']=vector_of_scans
'''

#code we're using atm to get indices to be kept
#test on 4 subjects
oa=np.ones(110968486, dtype=np.float32)
for filename in os.listdir('/media/james/ext4data1/current/projects/ramasubbu/vectortest'):
    with open('/media/james/ext4data1/current/projects/ramasubbu/vectortest/'+filename, 'rb') as f:
        vecs=pickle.load(f)
        for i in range(0,110968486):
            if (oa[i]!=0.0 and vecs[i]!=0.0):
                oa[i]=1.0
            else: oa[i]=0.0

#testing 1 subject
    with open('/media/james/ext4data1/current/projects/ramasubbu/vectors/c002.pickle', 'rb') as f:
        vecs=pickle.load(f)
        for i in range(0,110968486):
            if (oa[i]!=0.0 and vecs[i]!=0.0):
                oa[i]=1.0
            else: oa[i]=0.0

for i in range(len(vecs)):
     if oa[i]==1.0:
         a=a+1
         
'''
with open('/media/james/ext4data1/current/projects/ramasubbu/data.pickle', 'rb') as f: data= pickle.load(f)

X=np.array([data[i]['vector'] for i in data])

data=[]

with open('/media/james/ext4data1/current/projects/ramasubbu/X.pickle', 'wb') as d: pickle.dump(X, d, pickle.HIGHEST_PROTOCOL) 

#-- stop here and exit everything --

with open('/media/james/ext4data/current/projects/ramasubbu/X.pickle', 'rb') as f: X= pickle.load(f)

vt= sklearn.feature_selection.VarianceThreshold()

vt.fit(X)

nonzero_indices= vt.get_support(indices=True)
'''
