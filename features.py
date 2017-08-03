#!/usr/bin/env python3

from sklearn.feature_selection import SelectKBest
import copy, pickle
import numpy as np

#Run feature selection. Data here need to be transformed because they'll be used in the ML step.

def AllFeats():
    return

def SelKBest():
    with open('/media/james/ext4data1/current/projects/ramasubbu/data.pickle','rb') as f: data=pickle.load(f)
    with open('/media/james/ext4data1/current/projects/ramasubbu/inner_cv.pickle','rb') as f: inner_cv=pickle.load(f)       

#i is for the number of inner CV fold, j is for each subject in that fold.
    
    for i in range(25):
        X_train=[]
        print('fold {}/25'.format(i+1))
        for j in inner_cv['X_train'][i]:
            print('{}, {}'.format(j, len(X_train)))
            with open('/media/james/ext4data1/current/projects/ramasubbu/nonzerovectors/'+j+'.pickle','rb') as f: vecs=pickle.load(f)
            X_train=X_train+[vecs]
        
        X_test=[]
        for j in inner_cv['X_test'][i]:
            with open('/media/james/ext4data1/current/projects/ramasubbu/nonzerovectors/'+j+'.pickle','rb') as f: vecs=pickle.load(f)
            X_test=X_test+[vecs]
        
        X_train= np.array(X_train)
        y_train= np.array(inner_cv['y_train'][i])
         
        X_test=np.array(X_test)
        y_test=np.array(inner_cv['y_test'][i])
         
        print('\nPicking and Transforming Features\n')
        #So here, we have to get an index of the chosen features, then pare down the train AND test sets to just those features.
        skb= SelectKBest(k=20)  
        skb.fit(X_train, y_train)
        feats=skb.get_support(indices=True)
        print('Feature indices:\n{}'.format(feats))
        
        print('Transforming X_train')
        X_train=skb.transform(X_train)
        print('Transforming X_test')
        X_test=skb.transform(X_test)
        
        print('\nSaving Arrays\n')
            
        with open('/media/james/ext4data1/current/projects/ramasubbu/innercvfeatures/fold_'+str(i)+'_train.pickle','wb') as f: pickle.dump(X_train, f, pickle.HIGHEST_PROTOCOL)
        with open('/media/james/ext4data1/current/projects/ramasubbu/innercvfeatures/fold_'+str(i)+'_test.pickle','wb') as f: pickle.dump(X_test, f, pickle.HIGHEST_PROTOCOL) 
        with open('/media/james/ext4data1/current/projects/ramasubbu/innercvfeatures/fold_'+str(i)+'_feats.pickle','wb') as f: pickle.dump(feats, f, pickle.HIGHEST_PROTOCOL)  


'''        
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
