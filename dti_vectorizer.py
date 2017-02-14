#!/usr/bin/env python3

import os, csv, glob, ntpath, re, scipy.signal, pickle, gc
import pprint, random, glob, copy
import scipy.io as sio
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import stats
from nilearn.input_data import NiftiMasker
from nilearn.image import smooth_img
from sklearn import preprocessing

def SubjectInfo(): 
    '''Creates dict for scan data, incl file list & subject ID'''


    #FILES
    projectdir=input('Project directory (without quotes):')
    subdirs= glob.glob(projectdir + '*/')
    subdirs.sort()

    data_dict= {'directory': subdirs
                'subject ID': [], 
                'symptom severity': [],
                'treatment response': [],
                'feature vector': [],
                'data': []}
    
    with open('data_dict.pickle', 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)  
    
    #SUBJECT ID
    for i in data_dict['directory']:
        data_dict['subject ID'].append(ntpath.basename
            (os.path.splitext(i)[0]))
        
    #DATA
    mask = '/media/james/ext4data/current/projects/depression/07_Machine_Learning/02_FA_Skeletonized/mean_FA_skeleton_mask.nii.gz'
    nifti_masker= NiftiMasker(mask_img=mask)
    stdscaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
    
    imagefiles=[]
    for directory in data_dict['directory']
        for filename in glob.glob(directory + '/**/*.nii', recursive=True):
            imagefiles.append(filename)
            imagefiles.sort()
        for filename in imagefiles:
            image= nib.load(filename)
            img_data= image.get_data()
            img_data= np.concatenate(img_data) #3D-->2D
            img_data=np.concatenate(img_data) #2D--> vector
            
        result_img= smooth_img(a) #Makes a 4d concatenated image
        flat_img= nifti_masker.fit_transform(result_img)
        data_dict['data'].append(flat_img)
            
    
    #note: haven't done sklearn z-normalization yet- do with final multimodal array
        
    with open('data_dict.pickle', 'wb') as d:
        pickle.dump(data_dict, d, pickle.HIGHEST_PROTOCOL) 
