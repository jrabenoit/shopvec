#!/usr/bin/env python3 

from sklearn import decomposition
import copy, pickle
    
def NullTransform():
    return

def PCAchu():
    for i in range(10):
        with open('/media/james/ext4data1/current/projects/ramasubbu/innercvfeatures/fold_'+str(i)+'_train.pickle','rb') as f: X_train=pickle.load(f)
        with open('/media/james/ext4data1/current/projects/ramasubbu/innercvfeatures/fold_'+str(i)+'_test.pickle','rb') as f: X_test=pickle.load(f)
        #Reducing to 52 components so n_features<=n_components
        #trf= decomposition.PCA()
        trf= decomposition.PCA()
        trf.fit(X_train)
        X_train= trf.transform(X_train)
        X_test= trf.transform(X_test)
        
        with open('/media/james/ext4data1/current/projects/ramasubbu/pcafeatures/fold_'+str(i)+'_train.pickle','wb') as f: pickle.dump(X_train, f, pickle.HIGHEST_PROTOCOL)
        with open('/media/james/ext4data1/current/projects/ramasubbu/pcafeatures/fold_'+str(i)+'_test.pickle','wb') as f: pickle.dump(X_test, f, pickle.HIGHEST_PROTOCOL)

    return
