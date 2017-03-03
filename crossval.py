#!/usr/bin/env python3

import pprint, itertools, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import SelectKBest

def SelectGroups():   
    print('Symptom severity:\n')
    print('a0= control/mild\n a1= control/severe\n a2= control/very severe\n a3= mild/severe\n a4= mild/very severe\n a5= severe/very severe\n a6= control/all patients\n')
    print('Treatment response:\n')
    print('b0= control/non-responder\n b1= control/responder\n b2= control/remitter\n, b3= non-responder/responder\n b4= non-responder/remitter\n b5= responder/remitter\n b6= control/all patients')
    choice = int(input('Choice:'))
        
    if choice=a0: group=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,27,29,42,45,47,51]
    elif choice=a1: group=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,22,23,25,26,28,31,34,36,37,38,44,46,48,50,52]
    elif choice=a2: group=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,24,30,32,33,35,39,40,41,43,49]
    elif choice=a3: group=[18,19,20,21,27,29,42,45,47,51,22,23,25,26,28,31,34,36,37,38,44,46,48,50,52]
    elif choice=a4: group=[18,19,20,21,27,29,42,45,47,51,24,30,32,33,35,39,40,41,43,49]
    elif choice=a5: group=[22,23,25,26,28,31,34,36,37,38,44,46,48,50,52,24,30,32,33,35,39,40,41,43,49]
    elif choice=a6: group=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]
    elif choice=b0: group=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,21,22,23,26,31,34,37,38,46,33]
    elif choice=b1: group=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,29,42,47,51,25,36,48,52,35,39,40,43,49]
    elif choice=b2: group=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,27,45,28,44,50,24,30,32,41]
    elif choice=b3: group=[19,21,22,23,26,31,34,37,38,46,33,29,42,47,51,25,36,48,52,35,39,40,43,49]
    elif choice=b4: group=[19,21,22,23,26,31,34,37,38,46,33,18,20,27,45,28,44,50,24,30,32,41]
    elif choice=b5: group=[29,42,47,51,25,36,48,52,35,39,40,43,49,18,20,27,45,28,44,50,24,30,32,41]
    elif choice=b6: group=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52]

    with open('group_choice.pickle', 'wb') as f:
        pickle.dump(group, f, pickle.HIGHEST_PROTOCOL) 
        
    return

def OuterCv():        

    with open('group_choice.pickle', 'wb') as f:
        group= pickle.load(f)
        
    with open('data_dict.pickle', 'rb') as f:
        data_dict= pickle.load(f)
    
#    X= np.array([])
#    y= np.array([])
    #Does np.array need the [] inside ()?
    X=[]
    y=[]
    for i in group:
        X.extend(i)
        y.extend(data_dict['group type'][i])
    
    X_train, X_test, y_train, y_test= [], [], [], []      

    outer = StratifiedKFold(y, n_folds=5)
    for train_index, test_index in outer:
        X_train.append(X[train_index])
        X_test.append(X[test_index])
        y_train.append(y[train_index])
        y_test.append(y[test_index])
    
    outer_cv= {'X_train': np.array(X_train),
               'X_test': np.array(X_test),
               'y_train': np.array(y_train),
               'y_test': np.array(y_test)}

    with open('outer_cv.pickle', 'wb') as f:
        pickle.dump(outer_cv, f, pickle.HIGHEST_PROTOCOL) 

    return
    
def InnerCv():
    '''Set up as a flat structure of 25 df'''
    with open('pickles/outer_cv.pickle', 'rb') as f:
        outer_cv= pickle.load(f)
    
    X= outer_cv['X_train']
    y= outer_cv['y_train']
    
    X_train, X_test, y_train, y_test = [], [], [], []

    #read loop as, "for each pair of X and y lists in (X,y)"
    for X_, y_ in zip(X, y): 
        inner = StratifiedKFold(y_, n_folds=5)
        for train_index, test_index in inner:      
            X_train.append(X_[train_index])
            X_test.append(X_[test_index])
            y_train.append(y_[train_index])
            y_test.append(y_[test_index]) 

    inner_cv= {'X_train': X_train,
               'X_test': X_test,
               'y_train': y_train,
               'y_test': y_test}

    with open('pickles/inner_cv.pickle', 'wb') as f:
        pickle.dump(inner_cv, f, pickle.HIGHEST_PROTOCOL) 
    
    return

    
