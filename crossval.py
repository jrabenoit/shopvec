#!/usr/bin/env python3

import pprint, itertools, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold

def CVSetup():   
    
    with open('/media/james/ext4data/current/projects/ramasubbu/data_dict.pickle', 'rb') as f:
        data_dict= pickle.load(f)
    
    print('\nSymptom severity:\n')
    print(' 0= control/mild\n 1= control/severe\n 2= control/very severe\n 3= mild/severe\n 4= mild/very severe\n 5= severe/very severe\n 6= control/all patients\n')
    print('Treatment response:\n')
    print(' 7= control/non-responder\n 8= control/all responder\n 9= control/remitter only\n 10= non-responder/all responder\n 11= non-responder/remitter only\n 12= responder vs remitter\n 13= control/all patients')
    choice= int(input('\nChoice: '))
    
    if choice== 0: 
        group0= np.where(data_dict['symptom severity']==0)[0] 
        group1= np.where(data_dict['symptom severity']==1)[0]
        labels= np.array([[0]*len(group0)+[1]*len(group1)])[0]

    elif choice== 1: 
        group0= np.where(data_dict['symptom severity']==0)[0] 
        group1= np.where(data_dict['symptom severity']==2)[0]
        labels= np.array([[0]*len(group0)+[1]*len(group1)])[0]
        
    elif choice== 2:
        group0= np.where(data_dict['symptom severity']==0)[0] 
        group1= np.where(data_dict['symptom severity']==3)[0]
        labels= np.array([[0]*len(group0)+[1]*len(group1)])[0]
    
    elif choice== 3: 
        group0= np.where(data_dict['symptom severity']==1)[0] 
        group1= np.where(data_dict['symptom severity']==2)[0]
        labels= np.array([[0]*len(group0)+[1]*len(group1)])[0]
    
    elif choice== 4: 
        group0= np.where(data_dict['symptom severity']==1)[0] 
        group1= np.where(data_dict['symptom severity']==3)[0]
        labels= np.array([[0]*len(group0)+[1]*len(group1)])[0]
    
    elif choice== 5: 
        group0= np.where(data_dict['symptom severity']==2)[0] 
        group1= np.where(data_dict['symptom severity']==3)[0]
        labels= np.array([[0]*len(group0)+[1]*len(group1)])[0]
    
    elif choice== 6: 
        group0= np.where(data_dict['symptom severity']==0)[0]
        group1a= np.where(data_dict['symptom severity']==1)[0]
        group1b= np.where(data_dict['symptom severity']==2)[0]
        group1c= np.where(data_dict['symptom severity']==3)[0]
        group1= np.sort(np.append(group1a,np.append(group1b,group1c)))
        labels= np.array([[0]*len(group0)+[1]*len(group1)])[0]
    
    elif choice== 7:
        group0= np.where(data_dict['treatment response']==0)[0] 
        group1= np.where(data_dict['treatment response']==1)[0]
        labels= np.array([[0]*len(group0)+[1]*len(group1)])[0]
    
    elif choice== 8:
        group0= np.where(data_dict['treatment response']==0)[0] 
        group1a= np.where(data_dict['treatment response']==2)[0]
        group1b= np.where(data_dict['treatment response']==3)[0]
        group1= np.sort(np.append(group1a,group1b))
        labels= np.array([[0]*len(group0)+[1]*len(group1)])[0]
    
    elif choice== 9: 
        group0= np.where(data_dict['treatment response']==0)[0] 
        group1= np.where(data_dict['treatment response']==3)[0]
        labels= np.array([[0]*len(group0)+[1]*len(group1)])[0]
    
    elif choice== 10: 
        group0= np.where(data_dict['treatment response']==1)[0] 
        group1a= np.where(data_dict['treatment response']==2)[0]
        group1b= np.where(data_dict['treatment response']==3)[0]
        group1= np.sort(np.append(group1a,group1b))
        labels= np.array([[0]*len(group0)+[1]*len(group1)])[0]
    
    elif choice== 11:
        group0= np.where(data_dict['treatment response']==1)[0] 
        group1= np.where(data_dict['treatment response']==3)[0]
        labels= np.array([[0]*len(group0)+[1]*len(group1)])[0]
    
    elif choice== 12:
        group0= np.where(data_dict['treatment response']==2)[0] 
        group1= np.where(data_dict['treatment response']==3)[0]
        labels= np.array([[0]*len(group0)+[1]*len(group1)])[0]
    
    elif choice== 13:
        group0= np.where(data_dict['treatment response']==0)[0]
        group1a= np.where(data_dict['treatment response']==1)[0]
        group1b= np.where(data_dict['treatment response']==2)[0]
        group1c= np.where(data_dict['treatment response']==3)[0]
        group1= np.sort(np.append(group1a,np.append(group1b,group1c)))
        labels= np.array([[0]*len(group0)+[1]*len(group1)])[0]
    
    else: print('invalid selection.')
    
    outer_cv={'X_train': [], 'X_test': [],
              'y_train': [], 'y_test': []}

    X= np.append(group0,group1)
    y= labels
    
    X_train, X_test, y_train, y_test= [], [], [], []      

    skf = StratifiedKFold(n_splits=5)
    
    print('\nGroup size: {}'.format(len(X)))
    print('Group indices: {}'.format(X))
    print('\nLabel size: {}'.format(len(y)))
    print('Labels: {}\n'.format(y))
        
    for train_index, test_index in skf.split(X,y):
        #print("TRAIN:", train_index, "\nTEST:", test_index)
        X_train, X_test= X[train_index], X[test_index]
        y_train, y_test= y[train_index], y[test_index]
    
        outer_cv['X_train'].append(X_train)
        outer_cv['X_test'].append(X_test)
        outer_cv['y_train'].append(y_train)
        outer_cv['y_test'].append(y_test)

    with open('/media/james/ext4data/current/projects/ramasubbu/outer_cv.pickle', 'wb') as f:
        pickle.dump(outer_cv, f, pickle.HIGHEST_PROTOCOL) 

    return
    
def InnerCv():
    '''Set up as a flat structure of 25 df'''
    with open('/media/james/ext4data/current/projects/ramasubbu/outer_cv.pickle', 'rb') as f:
        outer_cv= pickle.load(f)

    inner_cv={'X_train': [], 'X_test': [],
              'y_train': [], 'y_test': []}
    
    X= outer_cv['X_train']
    y= outer_cv['y_train']
    
    X_train, X_test, y_train, y_test = [], [], [], []

    #read loop as, "for each pair of X and y lists in (X,y)"
    
    for X_, y_ in zip(X, y): 
        skf = StratifiedKFold(n_splits=5)
        for train_index, test_index in skf.split(X_,y_):      
            print("TRAIN:", train_index, "\nTEST:", test_index)
            X_train, X_test= X_[train_index], X_[test_index]
            y_train, y_test= y_[train_index], y_[test_index]

            inner_cv['X_train'].append(X_train)
            inner_cv['X_test'].append(X_test)
            inner_cv['y_train'].append(y_train)
            inner_cv['y_test'].append(y_test)

    with open('/media/james/ext4data/current/projects/ramasubbu/inner_cv.pickle', 'wb') as f:
        pickle.dump(inner_cv, f, pickle.HIGHEST_PROTOCOL) 
    
    return

    
