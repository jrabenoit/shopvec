#!/usr/bin/env python3

import scipy.io as sio
import numpy as np
import pandas as pd
import os, csv, glob, ntpath, re, scipy.signal, pickle, gc
from scipy import stats

def SubjectInfo(): 
    '''Creates dict for scan data, generates file list & subject ID'''
    filedir= input('File directory (without quotes):')
    files= glob.glob(filedir + '/*.mat')
    subject_id= list(range(len(files)))

    data_dict= {     'files': files,
                'subject_id': subject_id,
                  'subjects': [], 
                      'site': [],
                    'labels': [], 
                   'scan_tr': [],
                   'n_scans': [],
                  'scantime': [],
                   'samples': [],
                      'data': []}
    
    with open('pickles/data_dict.pickle', 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)  
    
    site_tr = {1:2, 2:2.3, 3:2, 4:2.52, 5:2.4, 6:1.25,\
               7:2, 8:2.4, 9:2.5, 10:2, 11:2}
    
    for i in data_dict['files']:
        data_dict['subjects'].append(ntpath.basename(os.path.splitext(i)[0]))
        if 'HC' in i: data_dict['labels'].append(0)
        elif 'MDD' in i: data_dict['labels'].append(1)
    
    for i in data_dict['subjects']:
        site= (list(map(int, re.findall('\d+',i)))[1])
        data_dict['site'].append(site)
        scan_tr= [v for k,v in site_tr.items() if k==site][0]
        data_dict['scan_tr'].append(scan_tr)       
 
    voxels= []
    for i in range(len(data_dict['files'])):
        voxels.append(sio.loadmat(data_dict['files'][i])['PAC_data'])

    for i in range(len(data_dict['files'])):
        data_dict['n_scans'].append(voxels[i].shape[1])
        #data_dict['scantime'].append(voxels[i].shape[1]*data_dict['scan_tr'][i])        
        data_dict['scantime'].append(270)
    
    for i in range(len(data_dict['files'])):
        n_samples= int(data_dict['scantime'][i]/1.25)
        data_dict['samples'].append(n_samples)

    with open('pickles/data_dict.pickle', 'wb') as d:
        pickle.dump(data_dict, d, pickle.HIGHEST_PROTOCOL) 

def MassageScans():
    with open('pickles/data_dict.pickle', 'rb') as f:
        data_dict= pickle.load(f)
    
    with open('mask_inds.mat','rb') as f:
        indices= sio.loadmat(f)['mask_inds']-1
    
    rows=[]
    for i in range(21260):
        rows.append(indices[i][0])
      
    for i in range(len(data_dict['files'])):
        voxels= []        
        subj= sio.loadmat(data_dict['files'][i])['PAC_data']
        for j in rows:
            voxels.append(subj[j])
        voxels= scipy.signal.resample(voxels, 60, axis=1)
        #voxels= scipy.signal.resample(voxels, data_dict['samples'][i], axis=1)
        #reshapes the numpy array from (21260,y) to (21260,105)
        #min_samples=min(data_dict['samples'])
        #voxels= voxels[:,:min_samples]
        data_dict['data'].append(voxels)

    for i in range(1,12,1):
        #list of subject ID's for site xx
        sitelist= [j for j in data_dict['subject_id']  if data_dict['site'][j]==i]
        datalist= np.array([data_dict['data'][i] for i in sitelist])
        #zdata= scipy.stats.zscore(datalist, axis=0,ddof=(len(sitelist)-1))
        #for i,j in zip(sitelist,datalist):
            #data_dict['data'][i]=j
            
    #for i in range(len(data_dict['files'])):   
    #    data_dict['data'][i]= np.reshape(data_dict['data'][i],(1,-1))
    
    #data_dict['data']= np.array([data_dict['data'][i][0] for i in range(len(data_dict['files']))])
    
    with open('pickles/data_dict.pickle', 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL) 
    
    return
