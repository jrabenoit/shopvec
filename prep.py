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

    #FILES
    projectdir=input('Project directory (without quotes):')
    subdirs= glob.glob(projectdir + '/*')
    subdirs.sort()

    data_dict= {'directory': subdirs,
                'subject ID': [], 
                'symptom severity': [],
                'treatment response': [],
                'data': []}
    
    #SUBJECT ID
    for i in data_dict['directory']:
        data_dict['subject ID'].append(ntpath.basename
            (os.path.splitext(i)[0]))
    
    #SYMPTOM SEVERITY
    #0=control, 1=mild-moderate, 2=severe, 3=very severe
    #control hamd <=7
    #mild-moderate hamd 14-19, severe 20-23, very severe 24+
    data_dict['symptom severity']= np.array([0,0,0,0,0,0,0,0,0,0,
                                    0,0,0,0,0,0,0,0,
                                    #1,
                                    1,1,1,2,2,3,2,2,1,
                                    2,1,3,2,3,3,2,3,2,2,
                                    2,3,3,3,1,3,2,1,2,1,
                                    2,3,2,1,2])
    #TREATMENT RESPONSE
    #control=0, non-response=1, response=2, remit=3
    #all remitters (3's) are responders (2's), but not vice versa
    data_dict['treatment response']= np.array([0,0,0,0,0,0,0,0,0,0,
                                      0,0,0,0,0,0,0,0,
                                      #3,
                                      1,3,1,1,1,3,2,1,3,
                                      3,2,3,1,3,1,1,2,2,1,
                                      1,2,2,3,2,2,3,3,1,2,
                                      2,2,3,2,2])
                                      
    #DATA
    mask = '/media/james/ext4data/current/projects/depression/07_Machine_Learning/02_FA_Skeletonized/mean_FA_skeleton_mask.nii.gz'
    nifti_masker= NiftiMasker(mask_img=mask)
    stdscaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
    
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
        concatimage=np.array(concatimage[0::10])
        print('vector length: {}'.format(len(concatimage)))
        #Feature vector concatenation, putting all scans together sequentially
        data_dict['data'].append(concatimage)
        print('subjects in data_dict: {}'.format(len(data_dict['data'])))
            
    
    #note: haven't done sklearn z-normalization yet- do with final multimodal array
        
    with open('/media/james/ext4data/current/projects/ramasubbu/data_dict.pickle', 'wb') as d:
        pickle.dump(data_dict, d, pickle.HIGHEST_PROTOCOL) 
        
#    with open('/media/james/ext4data/current/projects/ramasubbu/data_dict.pickle', 'rb') as f:
#        data_dict= pickle.load(f)
