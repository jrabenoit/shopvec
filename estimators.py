#!/usr/bin/env python3

import numpy as np
import pandas as pd
import copy, pickle
from sklearn import svm, naive_bayes, neighbors, ensemble, linear_model, tree, discriminant_analysis

def InnerFolds(group, run):
    with open('/media/james/ext4data1/current/projects/ramasubbu/inner_cv.pickle','rb') as f: cv=pickle.load(f)

    folds=10
    
    est= {'randomforest': ensemble.RandomForestClassifier(), 
          'extratrees': ensemble.ExtraTreesClassifier(),
          'kneighbors': neighbors.KNeighborsClassifier(),
          'naivebayes': naive_bayes.GaussianNB(),
          'decisiontree': tree.DecisionTreeClassifier(),
          'linearsvm': svm.LinearSVC(),
          'lda': discriminant_analysis.LinearDiscriminantAnalysis()
          }
   
    train_results={'fold':[], 'estimator':[], 'subjects':[], 'labels':[], 'predictions':[], 'scores':[], 'attempts':[]}
    test_results={'fold':[], 'estimator':[], 'subjects':[], 'labels':[], 'predictions':[], 'scores':[], 'attempts':[]}
   
    for i in range(folds):
        with open('/media/james/ext4data1/current/projects/ramasubbu/innercvfeatures/fold_'+str(i)+'_train.pickle','rb') as f: X_train=pickle.load(f)
        with open('/media/james/ext4data1/current/projects/ramasubbu/innercvfeatures/fold_'+str(i)+'_test.pickle','rb') as f: X_test=pickle.load(f)
        y_train= cv['y_train'][i]
        y_test= cv['y_test'][i]        
        
        print('\nFold {}/{}\n'.format(i+1, folds))   

        for j,k in zip(est.keys(), est.values()):
            k.fit(X_train, y_train)
            predictions= k.predict(X_train)
            scores= [1 if x==y else 0 for x,y in zip(cv['y_train'][i], predictions)]
            
            train_results['fold'].extend([i+1]*len(X_train))
            train_results['estimator'].extend([j]*len(X_train))
            train_results['subjects'].extend(cv['X_train'][i])
            train_results['labels'].extend(cv['y_train'][i])
            train_results['predictions'].extend(predictions)
            train_results['scores'].extend(scores)
            train_results['attempts'].extend([1]*len(X_train))
        
        for j,k in zip(est.keys(), est.values()):    
            k.fit(X_test, y_test)
            predictions= k.predict(X_test)
            scores= [1 if x==y else 0 for x,y in zip(cv['y_test'][i], predictions)]
            
            test_results['fold'].extend([i+1]*len(X_test))
            test_results['estimator'].extend([j]*len(X_test))
            test_results['subjects'].extend(cv['X_test'][i])
            test_results['labels'].extend(cv['y_test'][i])
            test_results['predictions'].extend(predictions)
            test_results['scores'].extend(scores)
            test_results['attempts'].extend([1]*len(X_test))

    train_results=pd.DataFrame(train_results)
    test_results=pd.DataFrame(test_results)
    
    with open('/media/james/ext4data1/current/projects/ramasubbu/inner_train_results_group_'+str(group)+'_run_'+str(run)+'.pickle', 'wb') as d: pickle.dump(train_results, d, pickle.HIGHEST_PROTOCOL) 
    with open('/media/james/ext4data1/current/projects/ramasubbu/inner_test_results_group_'+str(group)+'_run_'+str(run)+'.pickle', 'wb') as d: pickle.dump(test_results, d, pickle.HIGHEST_PROTOCOL) 

    return

def OuterFolds(group, run):
    with open('/media/james/ext4data1/current/projects/ramasubbu/outer_cv.pickle','rb') as f: cv=pickle.load(f)

    folds=5
    
    est= {'randomforest': ensemble.RandomForestClassifier(), 
          'extratrees': ensemble.ExtraTreesClassifier(),
          'kneighbors': neighbors.KNeighborsClassifier(),
          'naivebayes': naive_bayes.GaussianNB(),
          'decisiontree': tree.DecisionTreeClassifier(),
          'linearsvm': svm.LinearSVC(),
          'lda': discriminant_analysis.LinearDiscriminantAnalysis()
          }
   
    train_results={'fold':[], 'estimator':[], 'subjects':[], 'labels':[], 'predictions':[], 'scores':[]}
    test_results={'fold':[], 'estimator':[], 'subjects':[], 'labels':[], 'predictions':[], 'scores':[]}
   
    for i in range(folds):
        with open('/media/james/ext4data1/current/projects/ramasubbu/outercvfeatures/fold_'+str(i)+'_train.pickle','rb') as f: X_train=pickle.load(f)
        with open('/media/james/ext4data1/current/projects/ramasubbu/outercvfeatures/fold_'+str(i)+'_test.pickle','rb') as f: X_test=pickle.load(f)
        y_train= cv['y_train'][i]
        y_test= cv['y_test'][i]        
        
        print('\nFold {}/{}\n'.format(i+1, folds))   

        for j,k in zip(est.keys(), est.values()):
            k.fit(X_train, y_train)
            fold=[str(i+1)]*len(cv['X_train'][i])  
            estimator=[j]*len(cv['X_train'][i])
            subjects= cv['X_train'][i]
            labels= cv['y_train'][i]
            predictions= k.predict(X_train)
            scores= [1 if x==y else 0 for x,y in zip(labels, predictions)]
            train_results['fold'].extend(fold)
            train_results['estimator'].extend(estimator)
            train_results['subjects'].extend(subjects)
            train_results['labels'].extend(labels)
            train_results['predictions'].extend(predictions)
            train_results['scores'].extend(scores)
        
        for j,k in zip(est.keys(), est.values()):    
            k.fit(X_test, y_test)
            fold=[str(i+1)]*len(cv['X_test'][i])  
            estimator=[j]*len(cv['X_test'][i])
            subjects= cv['X_test'][i]
            labels= cv['y_test'][i]
            predictions= k.predict(X_test)
            scores= [1 if x==y else 0 for x,y in zip(labels, predictions)]
            test_results['fold'].extend(fold)
            test_results['estimator'].extend(estimator)
            test_results['subjects'].extend(subjects)
            test_results['labels'].extend(labels)
            test_results['predictions'].extend(predictions)
            test_results['scores'].extend(scores)                   
    
    train_results=pd.DataFrame(train_results)
    test_results=pd.DataFrame(test_results)
    
    with open('/media/james/ext4data1/current/projects/ramasubbu/outer_train_results_group_'+str(group)+'_run_'+str(run)+'.pickle', 'wb') as d: pickle.dump(train_results, d, pickle.HIGHEST_PROTOCOL) 
    with open('/media/james/ext4data1/current/projects/ramasubbu/outer_test_results_group_'+str(group)+'_run_'+str(run)+'.pickle', 'wb') as d: pickle.dump(test_results, d, pickle.HIGHEST_PROTOCOL) 

    return
