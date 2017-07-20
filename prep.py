#!/usr/bin/env python3

import os, csv, glob, ntpath, re, scipy.signal, pickle, gc, pprint, random, glob, copy
import scipy.io as sio
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats, signal
from nilearn.input_data import NiftiMasker
from nilearn.image import smooth_img
from sklearn import preprocessing

def SubjectInfo(): 
    '''Creates dict for scan data, incl file list & subject ID'''
    
    #FILE LIST
    projectdir=input('Project directory (without quotes):')
    subdirs= glob.glob(projectdir + '/*')
    subdirs.sort()
    
    #SUBJECT ID LIST
    subjects=[]
    for i in subdirs:
        subjects.append(ntpath.basename
            (os.path.splitext(i)[0]))

    #TOP LEVEL DICT
    data={}
    
    #SUB-DICTS KEYED TO SUBJECT ID
    for i in subjects:
        data[i]={}
    
    #FILE LOCATION PER SUBJECT    
    for i,j in zip(subjects,subdirs):
        data[i]['dir']=j
    
    #SYMPTOM SEVERITY
    sx= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
         1,1,1,2,2,3,2,2,1,2,1,3,2,3,3,2,3,2,2,2,3,3,3,1,3,2,1,2,1,2,3,2,1,2]
    for i,j in zip(subjects, sx):
        data[i]['sx severity']= j
    
    #TREATMENT RESPONSE
    tx= [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
         1,3,1,1,1,3,2,1,3,3,2,3,1,3,1,1,2,2,1,1,2,2,3,2,2,3,3,1,2,2,2,3,2,2]
    for i,j in zip(subjects, tx):
        data[i]['tx response']= j
                                      
    #CONCATENATED SCAN VECTORS
    mask = '/media/james/ext4data/current/projects/depression/07_Machine_Learning/02_FA_Skeletonized/mean_FA_skeleton_mask.nii.gz'
    nifti_masker= NiftiMasker(mask_img=mask)
    stdscaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
    
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
        #data[subject]['vector']=vector_of_scans
        with open('/media/james/ext4data/current/projects/ramasubbu/vectors/'+subject+'.pickle', 'wb') as d:
            pickle.dump(vector_of_scans, d, pickle.HIGHEST_PROTOCOL) 
    
    
    #with open('/media/james/ext4data/current/projects/ramasubbu/data.pickle', 'wb') as d:
    #    pickle.dump(data, d, pickle.HIGHEST_PROTOCOL) 

#note: haven't done sklearn z-normalization yet- do with final multimodal array
#    with open('/media/james/ext4data/current/projects/ramasubbu/data.pickle', 'rb') as f:
#        data= pickle.load(f)   
