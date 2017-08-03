#!/usr/bin/env python3

import numpy as np
import copy, pickle
from sklearn import svm, naive_bayes, neighbors, ensemble, linear_model

# .fit fits the model to the dataset in brackets. 
# .score tests the fitted model on data.
# .predict gives the per-subject predictions

def GauNaiBay():
    
    with open('/media/james/ext4data1/current/projects/ramasubbu/inner_cv.pickle','rb') as f: inner_cv=pickle.load(f)       
    
    for i in range(1):
        with open('/media/james/ext4data1/current/projects/ramasubbu/innercvfeatures/fold_'+str(i)+'_train.pickle','rb') as f:    
            X_train=pickle.load(f)
        with open('/media/james/ext4data1/current/projects/ramasubbu/innercvfeatures/fold_'+str(i)+'_test.pickle','rb') as f:
            X_test=pickle.load(f)
        
        y_train= inner_cv['y_train'][i]
        y_test= inner_cv['y_test'][i]
        
        print('X_train\n{}\n'.format(X_train[0:1:1]))
        print('X_test\n{}\n'.format(X_test[0:1:1]))
        print('y_train\n{}\n'.format(y_train))
        print('y_test\n{}\n'.format(y_test))
        
        gnb= naive_bayes.GaussianNB()        
        gnb.fit(X_train, y_train)
        
        #make a dict
        #outer dict: subject name
        #list of inner dicts: result for one thing e.g. 
        #subject(outer loop 3(inner loop 4(prediction, label, etc)))))
        
        train_scores= gnb.score(X_train, y_train)
        print(train_scores)
        test_scores= gnb.score(X_test, y_test)
        print(test_scores)
    
    return

'''  
def KNeighbors(X_train, X_test, y_train, y_test):
    for i in range(0,len(X_train)):
        knc = neighbors.KNeighborsClassifier()
        knc.fit(X_train[i], y_train[i])
        X_train[i] = knc.score(X_train[i], y_train[i])
        X_test[i] = knc.score(X_test[i], y_test[i])
    return X_train, X_test

def CSupSvc():
    with open('data_dict_vectors.pickle','rb') as f:
        data_dict=pickle.load(f)
    with open('inner_cv.pickle','rb') as f:
        inner_cv=pickle.load(f) 
    
    scores= {'train': [], 'test': []}
    for i in range(25):
        X_train= np.array([data_dict['data'][inner_cv['X_train'][i][j]] for j in range(len(inner_cv['X_train'][i]))])
        X_test= np.array([data_dict['data'][inner_cv['X_test'][i][j]] for j in range(len(inner_cv['X_test'][i]))])
        y_train= inner_cv['y_train'][i]
        y_test= inner_cv['y_test'][i]
        est = svm.SVC()
        est.fit(X_train, y_train)
        scores['train'].append(est.score(X_train, y_train))
        scores['test'].append(est.score(X_test, y_test))
    
    with open('csvc_scores.pickle','wb') as f:
        pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL) 

    return 
    
def RandomForest():
    with open('data_dict.pickle','rb') as f:
        data_dict=pickle.load(f)
    with open('inner_cv.pickle','rb') as f:
        inner_cv=pickle.load(f) 
    
    scores= {'train': [], 'test': []}
    for i in range(25):
        X_train= np.array([data_dict['data'][inner_cv['X_train'][i][j]]\
                          for j in range(len(inner_cv['X_train'][i]))])
        X_test= np.array([data_dict['data'][inner_cv['X_test'][i][j]]\
                         for j in range(len(inner_cv['X_test'][i]))])
        y_train= inner_cv['y_train'][i]
        y_test= inner_cv['y_test'][i]

        est = ensemble.RandomForestClassifier()
        est.fit(X_train, y_train)
        scores['train'].append(est.score(X_train, y_train))
        scores['test'].append(est.score(X_test, y_test))
    
    with open('rf_scores.pickle','wb') as f:
        pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL) 

    return

def ExtraTrees(X_train, X_test, y_train, y_test):
    for i in range(len(X_train)):
        rf = ensemble.ExtraTreesClassifier()
        rf.fit(X_train[i], y_train[i])
        X_train[i] = rf.score(X_train[i], y_train[i])
        X_test[i] = rf.score(X_test[i], y_test[i])
    return X_train, X_test

def LinearSgd(X_train, X_test, y_train, y_test):
    for i in range(len(X_train)):
        sgd = linear_model.SGDClassifier()
        sgd.fit(X_train[i], y_train[i])
        X_train[i] = sgd.score(X_train[i], y_train[i])
        X_test[i] = sgd.score(X_test[i], y_test[i])
    return X_train, X_test
'''    
