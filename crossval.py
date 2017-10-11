#!/usr/bin/env python3

import pprint, itertools, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold

def OuterCV():   
    
    with open('/media/james/ext4data1/current/projects/ramasubbu/data.pickle', 'rb') as f:
        data= pickle.load(f)
    
    print('\nSymptom Severity:\n')
    print(' 0= control/mild\n 1= control/severe\n 2= control/very severe\n 3= mild/severe\n 4= mild/very severe\n 5= severe/very severe\n 6= control/all patients\n')
    
    print('Treatment Response:\n')
    print(' 7= control/non-responder\n 8= control/all responder\n 9= control/remitter only\n 10= non-responder/all responder\n 11= non-responder/remitter only\n 12= responder vs remitter\n 13= control/all patients')
    
    print('\nStudy contains {} Subjects'.format(len(data)))
    choice= int(input('\nChoice: '))
        
    group0=[]
    group1=[]
    if choice== 0:
        for i in data:
            if data[i]['sx severity']==0: group0.append(i)
            if data[i]['sx severity']==1: group1.append(i)
    elif choice== 1:
        for i in data:
            if data[i]['sx severity']==0: group0.append(i)
            if data[i]['sx severity']==2: group1.append(i)
    elif choice== 2:
        for i in data:
            if data[i]['sx severity']==0: group0.append(i)
            if data[i]['sx severity']==3: group1.append(i)   
    elif choice== 3:
        for i in data:
            if data[i]['sx severity']==1: group0.append(i)
            if data[i]['sx severity']==2: group1.append(i)       
    elif choice== 4:
        for i in data:
            if data[i]['sx severity']==1: group0.append(i)
            if data[i]['sx severity']==3: group1.append(i)      
    elif choice== 5:
        for i in data:
            if data[i]['sx severity']==2: group0.append(i)
            if data[i]['sx severity']==3: group1.append(i)
    elif choice== 6:
        group1a=[]
        group1b=[]
        group1c=[]
        for i in data:
            if data[i]['sx severity']==0: group0.append(i)
            if data[i]['sx severity']==1: group1a.append(i)
            if data[i]['sx severity']==2: group1b.append(i)
            if data[i]['sx severity']==3: group1c.append(i)
            group1= np.sort(np.append(group1a,np.append(group1b,group1c)))   
    elif choice== 7:
        for i in data:
            if data[i]['tx response']==0: group0.append(i)
            if data[i]['tx response']==1: group1.append(i)
    elif choice== 8:
        group1a=[]
        group1b=[]
        for i in data:
            if data[i]['tx response']==0: group0.append(i)
            if data[i]['tx response']==2: group1a.append(i)
            if data[i]['tx response']==3: group1a.append(i)
            group1= np.sort(np.append(group1a,group1b))  
    elif choice== 9:
        for i in data:
            if data[i]['tx response']==0: group0.append(i)
            if data[i]['tx response']==3: group1.append(i)
    elif choice== 10:
        group1a=[]
        group1b=[]
        for i in data:
            if data[i]['tx response']==1: group0.append(i)
            if data[i]['tx response']==2: group1a.append(i)
            if data[i]['tx response']==3: group1a.append(i)
            group1= np.sort(np.append(group1a,group1b))      
    elif choice== 11:
        for i in data:
            if data[i]['tx response']==1: group0.append(i)
            if data[i]['tx response']==3: group1.append(i)
    elif choice== 12:
        for i in data:
            if data[i]['tx response']==2: group0.append(i)
            if data[i]['tx response']==3: group1.append(i)  
    elif choice== 13:
        group1a=[]
        group1b=[]
        group1c=[]
        for i in data:
            if data[i]['tx response']==0: group0.append(i)
            if data[i]['tx response']==1: group1a.append(i)
            if data[i]['tx response']==2: group1a.append(i)
            if data[i]['tx response']==3: group1a.append(i)
            group1= np.sort(np.append(group1a,np.append(group1b,group1c)))
    else: print('Invalid selection.')
    
    group0.sort()
    group1.sort()
    
    if len(group0)>len(group1): 
        group0=np.random.choice(group0, size=len(group1), replace= False) 
        labels= np.array([0]*len(group0)+[1]*len(group1)) 
    elif len(group0)<len(group1): 
        group1=np.random.choice(group1, size=len(group0), replace= False)
        labels= np.array([0]*len(group0)+[1]*len(group1)) 
    
    #For group 0 and 1, take n random subjects in the larger group where n=n_subjects in smaller group.    
    
    outer_cv={'X_train': [], 'X_test': [],
              'y_train': [], 'y_test': []}

    X= np.append(group0,group1)
    y= labels
    
    X_train, X_test, y_train, y_test= [], [], [], []      

    skf = StratifiedKFold(n_splits=5)
    
    print('\nGroup size: {}, group0: {}, group1: {}'.format(len(X), len(group0), len(group1)))
    print('Group indices: {}'.format(X))
    print('\nLabel size: {}, group0: {}, group1: {}'.format(len(y), (y==0).sum(), (y==1).sum()))
    print('Labels: {}\n'.format(y))
        
    for train_index, test_index in skf.split(X,y):
        #print("TRAIN:", train_index, "\nTEST:", test_index)
        X_train, X_test= X[train_index], X[test_index]
        y_train, y_test= y[train_index], y[test_index]
    
        outer_cv['X_train'].append(X_train)
        outer_cv['X_test'].append(X_test)
        outer_cv['y_train'].append(y_train)
        outer_cv['y_test'].append(y_test)

    with open('/media/james/ext4data1/current/projects/ramasubbu/outer_cv.pickle', 'wb') as f: pickle.dump(outer_cv, f, pickle.HIGHEST_PROTOCOL) 

    return
    
def InnerCV():
    '''Set up as a flat structure of 25 df'''
    with open('/media/james/ext4data1/current/projects/ramasubbu/outer_cv.pickle', 'rb') as f: outer_cv= pickle.load(f)

    inner_cv={'X_train': [], 'X_test': [],
              'y_train': [], 'y_test': []}
    
    X= outer_cv['X_train']
    y= outer_cv['y_train']
    
    X_train, X_test, y_train, y_test = [], [], [], []
    
    #change X to subjects, y to labels
    #read loop as, "for each pair of X and y lists in (X,y)"
    
    for X_, y_ in zip(X, y): 
        skf = StratifiedKFold(n_splits=2)
        for train_index, test_index in skf.split(X_,y_):      
            X_train, X_test= X_[train_index], X_[test_index]
            y_train, y_test= y_[train_index], y_[test_index]

            inner_cv['X_train'].append(X_train)
            inner_cv['X_test'].append(X_test)
            inner_cv['y_train'].append(y_train)
            inner_cv['y_test'].append(y_test)

    with open('/media/james/ext4data1/current/projects/ramasubbu/inner_cv.pickle', 'wb') as f:
        pickle.dump(inner_cv, f, pickle.HIGHEST_PROTOCOL) 
    
    return

    
