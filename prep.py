#!/usr/bin/env python3

import os, csv, glob, ntpath, re, scipy.signal, pickle, gc
import pprint, random, glob, copy
import scipy.io as sio
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from scipy import signal
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
        data[i]['tx severity']= j
                                      
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
        data[subject]['vector']=vector_of_scans
    
    with open('/media/james/ext4data/current/projects/ramasubbu/data.pickle', 'wb') as d:
        pickle.dump(data, d, pickle.HIGHEST_PROTOCOL) 

'''        
    for directory, subject in zip(data_dict['directory'], data_dict['subject ID']):
        #Creates list of .nii files in directory
        print('\nSubject ID: {}'.format(subject))
        imagefiles=[]
        for filename in glob.glob(directory + '/**/*.nii', recursive=True):
            imagefiles.append(filename)
            imagefiles.sort()
        print('# of scans: {}'.format(len(imagefiles)))
        #Turns each nii file into a vector
        concatimage= []
        for filename in imagefiles:
            image= nib.load(filename)
            img_data= image.get_data()
            img_data= np.reshape(img_data, (1,-1))
            img_data= np.concatenate(img_data)
            concatimage.extend(img_data)
        concatimage=np.array(concatimage[0::])
        print('vector length: {}'.format(len(concatimage)))
        #Feature vector concatenation, putting all scans together sequentially
        data_dict['data'].append(concatimage)
        print('subjects in data_dict: {}'.format(len(data_dict['data'])))
'''            
    
    #note: haven't done sklearn z-normalization yet- do with final multimodal array
        

        
#    with open('/media/james/ext4data/current/projects/ramasubbu/data_dict.pickle', 'rb') as f:
#        data_dict= pickle.load(f)
