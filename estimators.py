#!/usr/bin/env python3

import numpy as np
import copy, pickle
from sklearn import svm, naive_bayes, neighbors, ensemble, linear_model

# .fit fits the model to the dataset in brackets. 
# .score tests the fitted model on data.
# .predict gives the per-subject predictions

def GauNaiBay(trf):
    with open('pickles/data_dict_vectors.pickle','rb') as f:
        data_dict=pickle.load(f)
    with open('pickles/inner_cv.pickle','rb') as f:
        inner_cv=pickle.load(f) 
    
    scores= {'train': [], 'test': []}
    for i in range(25):
        X_train= np.array([data_dict['data'][inner_cv['X_train'][i][j]]\
                          for j in range(len(inner_cv['X_train'][i]))])
        X_test= np.array([data_dict['data'][inner_cv['X_test'][i][j]]\
                         for j in range(len(inner_cv['X_test'][i]))])
        y_train= inner_cv['y_train'][i]
        y_test= inner_cv['y_test'][i]
        
        est = naive_bayes.GaussianNB()
        est.fit(X_train, y_train)
        scores['train'].append(est.score(X_train, y_train))
        scores['test'].append(est.score(X_test, y_test))
    
    with open('pickles/gnb_scores.pickle','wb') as f:
        pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL) 

    return

'''  
def KNeighbors(X_train, X_test, y_train, y_test):
    for i in range(0,len(X_train)):
        knc = neighbors.KNeighborsClassifier()
        knc.fit(X_train[i], y_train[i])
        X_train[i] = knc.score(X_train[i], y_train[i])
        X_test[i] = knc.score(X_test[i], y_test[i])
    return X_train, X_test
'''
def CSupSvc():
    with open('pickles/data_dict_vectors.pickle','rb') as f:
        data_dict=pickle.load(f)
    with open('pickles/inner_cv.pickle','rb') as f:
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
    
    with open('pickles/csvc_scores.pickle','wb') as f:
        pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL) 

    return 
    
def RandomForest():
    with open('pickles/data_dict.pickle','rb') as f:
        data_dict=pickle.load(f)
    with open('pickles/inner_cv.pickle','rb') as f:
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
    
    with open('pickles/rf_scores.pickle','wb') as f:
        pickle.dump(scores, f, pickle.HIGHEST_PROTOCOL) 

    return

'''
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
