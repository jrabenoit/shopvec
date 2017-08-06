#!/usr/bin/env python3 

from sklearn import decomposition
import copy, pickle
    
def NullTransform():
    return

def FastICA():
    for i in range(25):
        with open('/media/james/ext4data1/current/projects/ramasubbu/innercvfeatures/fold_'+str(i)+'_train.pickle','rb') as f: X_train=pickle.load(f)
        with open('/media/james/ext4data1/current/projects/ramasubbu/innercvfeatures/fold_'+str(i)+'_test.pickle','rb') as f: X_test=pickle.load(f)
        trf= decomposition.FastICA(n_components=2)
        trf.fit(X_train)
        X_train= trf.transform(X_train)
        X_test= trf.transform(X_test)
        
        with open('/media/james/ext4data1/current/projects/ramasubbu/fasticafeatures/fold_'+str(i)+'_train.pickle','wb') as f: pickle.dump(X_train, f, pickle.HIGHEST_PROTOCOL)
        with open('/media/james/ext4data1/current/projects/ramasubbu/fasticafeatures/fold_'+str(i)+'_test.pickle','wb') as f: pickle.dump(X_test, f, pickle.HIGHEST_PROTOCOL)

'''
def FastIca():
    #ICA on whole dataset... constrain to each inner loop training set next
    with open('pickles/data_dict.pickle','rb') as f:
        data_dict=pickle.load(f)
    
    ica= {}
    ica['data']= []
    #scores= {'train': [], 'test': []}
    X= data_dict['data']
    trf= decomposition.FastICA(n_components=2)
    ica['data'].append(trf.fit_transform(X))        
    
    with open('pickles/ica_data.pickle','wb') as f:
        pickle.dump(ica, f, pickle.HIGHEST_PROTOCOL) 
    
    return
       
def RPca(iX_train, iX_test, iy_train, iy_test, n_components=3):
    dX_train = copy.copy(iX_train)
    dX_test = copy.copy(iX_test)
    dy_train = copy.copy(iy_train)
    dy_test = copy.copy(iy_test)
    for i in range(0,len(iX_train)):
        pca = decomposition.RandomizedPCA(n_components=n_components)
        pca.fit(dX_train[i])
        dX_train[i] = pca.transform(dX_train[i])
        dX_test[i] = pca.transform(dX_test[i])
    return dX_train, dX_test, dy_train, dy_test
'''
