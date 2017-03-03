#!/usr/bin/env python3

import scipy.io as sio
import numpy as np
import pandas as pd
import os, csv, glob, ntpath, re, scipy.signal, scipy.stats, pickle, gc
from sklearn.preprocessing import StandardScaler
from scipy import stats
    
def VectorizeMeCaptain():
    with open('pickles/data_dict.pickle', 'rb') as f:
        data_dict= pickle.load(f)
    
    indices= sio.loadmat('/home/james/github/marvin/mask_inds.mat')['mask_inds']-1
    
    rows=[]
    for i in range(21260):
        rows.append(indices[i][0])
      
    for i in range(334):
        voxels= []        
        subj= sio.loadmat(data_dict['files'][i])['PAC_data']
        for j in rows:
            voxels.append(subj[j])
        voxels= scipy.signal.resample(voxels, data_dict['samples'][i], axis=1)
        voxels= voxels[:,:105]
    

    for i in range(1,12,1):
        #list of subject ID's for site xx
        sitelist= [j for j in data_dict['subject_id']  if data_dict['site'][j]==i]
        datalist= np.array([data_dict['data'][i] for i in sitelist])
        zdata= scipy.stats.zscore(datalist, axis=0,ddof=(len(sitelist)-1))
        for i,j in zip(sitelist,zdata):
            data_dict['data'][i]=j
            
    for i in range(334):   
        voxels= np.reshape(voxels,(1,-1))
        data_dict['data'].append(voxels)

    with open('pickles/data_dict_matrices.pickle', 'wb') as d:
        pickle.dump(data_dict, d, pickle.HIGHEST_PROTOCOL) 
    
    return
    
