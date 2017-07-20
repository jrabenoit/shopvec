#!/usr/bin/env python3

from sklearn.feature_selection import SelectKBest
import copy, pickle
import numpy as np

#Run feature selection. Data here need to be transformed because they'll be used in the ML step.

def AllFeats():
    return

def SelKBest():
    with open('/media/james/ext4data/current/projects/ramasubbu/data.pickle','rb') as f:
        data=pickle.load(f)
    with open('/media/james/ext4data/current/projects/ramasubbu/inner_cv.pickle','rb') as f:
        inner_cv=pickle.load(f)       
    
    features={}
    for i in range(25):
        X_train= np.array([data[j]['vector'] for j in inner_cv['X_train'][i]])
        X_test= np.array([data[j]['vector'] for j in inner_cv['X_test'][i]])
        y_train= inner_cv['y_train'][i]
        y_test= inner_cv['y_test'][i]
        
        print(X_train)
        print(X_test)
        print(y_train)
        print(y_test)
        
        skb= SelectKBest(k=20)        
        skb.fit_transform(X_train, y_train)
        
        k= 'fold ' + str(i)
        features[k]= X_train
    
    with open('/media/james/ext4data/current/projects/ramasubbu/features.pickle','wb') as f:
        pickle.dump(features, f, pickle.HIGHEST_PROTOCOL) 
    
    return
'''   
def SelKBest(X_train, X_test, y_train, y_test, k=20):
        with open('/media/james/ext4data/current/projects/ramasubbu/data_dict.pickle','rb') as f:
        data_dict=pickle.load(f)
    with open('/media/james/ext4data/current/projects/ramasubbu/inner_cv.pickle','rb') as f:
        inner_cv=pickle.load(f)  

    skb = SelectKBest(f_classif, k=k)
    for i in range(25):
        X_train= np.array([data_dict['data'][j] for j in inner_cv['X_train'][i]])
        X_test= np.array([data_dict['data'][j] for j in inner_cv['X_test'][i]])
        y_train= inner_cv['y_train'][i]
        y_test= inner_cv['y_test'][i]
    
        skb = SelectKBest(f_classif, k=k)
        skb.fit(fX_train, fy_train)
        fX_train = skb.transform(fX_train)
        fX_test = skb.transform(fX_test)

    with open('/media/james/ext4data/current/projects/ramasubbu/data_dict.pickle','wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL) 
        
    return
'''
