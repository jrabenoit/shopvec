#!/usr/bin/env python3

import numpy as np
import copy, pickle
from sklearn import svm, naive_bayes, neighbors, ensemble, linear_model

# .fit fits the model to the dataset in brackets. 
# .score tests the fitted model on data.
# .predict gives the per-subject predictions

#To open score dict for gnb: with open('/media/james/ext4data1/current/projects/ramasubbu/gnbscores.pickle','rb') as f: scores=pickle.load(f)

def GauNaiBay():
    with open('/media/james/ext4data1/current/projects/ramasubbu/inner_cv.pickle','rb') as f: inner_cv=pickle.load(f)       
    
    scores={'fold': [], 'inner train':[], 'inner test':[]}
    
    for i in range(25):
        with open('/media/james/ext4data1/current/projects/ramasubbu/fasticafeatures/fold_'+str(i)+'_train.pickle','rb') as f:    
            X_train=pickle.load(f)
        with open('/media/james/ext4data1/current/projects/ramasubbu/fasticafeatures/fold_'+str(i)+'_test.pickle','rb') as f:
            X_test=pickle.load(f)
        
        y_train= inner_cv['y_train'][i]
        y_test= inner_cv['y_test'][i]
        
        est= naive_bayes.GaussianNB()
        est.fit(X_train, y_train)
                
        print('\nFold {}/25\n'.format((i+1)))
        
        predicted_train= est.predict(X_train)
        train_score= est.score(X_train, y_train)
        print('X_train predictions: {}'.format(predicted_train))
        print('y_train actual vals: {}'.format(y_train))        
        print('Training set score: {}%\n'.format((train_score*100)))
        
        predicted_test= est.predict(X_test)
        test_score= est.score(X_test, y_test)
        print('X_test predictions: {}'.format(predicted_test))
        print('y_test actual vals: {}'.format(y_test))
        print('Test set score: {}%\n'.format((test_score*100)))

        scores['fold'].append(i)
        scores['inner train'].append(train_score)
        scores['inner test'].append(test_score)
    
    with open('/media/james/ext4data1/current/projects/ramasubbu/gnbscores.pickle', 'wb') as d: pickle.dump(scores, d, pickle.HIGHEST_PROTOCOL) 

    return

def RandomForest():
    with open('/media/james/ext4data1/current/projects/ramasubbu/inner_cv.pickle','rb') as f: inner_cv=pickle.load(f)       
    
    scores={'fold': [], 'inner train':[], 'inner test':[]}
    
    for i in range(25):
        with open('/media/james/ext4data1/current/projects/ramasubbu/fasticafeatures/fold_'+str(i)+'_train.pickle','rb') as f:    
            X_train=pickle.load(f)
        with open('/media/james/ext4data1/current/projects/ramasubbu/fasticafeatures/fold_'+str(i)+'_test.pickle','rb') as f:
            X_test=pickle.load(f)
        
        y_train= inner_cv['y_train'][i]
        y_test= inner_cv['y_test'][i]
        
        est= ensemble.RandomForestClassifier()
        est.fit(X_train, y_train)
                
        print('\nFold {}/25\n'.format((i+1)))
        
        predicted_train= est.predict(X_train)
        train_score= est.score(X_train, y_train)
        print('X_train predictions: {}'.format(predicted_train))
        print('y_train actual vals: {}'.format(y_train))        
        print('Training set score: {}%\n'.format((train_score*100)))
        
        predicted_test= est.predict(X_test)
        test_score= est.score(X_test, y_test)
        print('X_test predictions: {}'.format(predicted_test))
        print('y_test actual vals: {}'.format(y_test))
        print('Test set score: {}%\n'.format((test_score*100)))

        scores['fold'].append(i)
        scores['inner train'].append(train_score)
        scores['inner test'].append(test_score)
        
    
    with open('/media/james/ext4data1/current/projects/ramasubbu/rfscores.pickle', 'wb') as d: pickle.dump(scores, d, pickle.HIGHEST_PROTOCOL) 

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
