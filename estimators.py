#!/usr/bin/env python3

import numpy as np
import copy, pickle
from sklearn import svm, naive_bayes, neighbors, ensemble, linear_model, tree, discriminant_analysis

def InnerFolds():
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
   
    train_results={'fold':[], 'estimator':[], 'subjects':[], 'labels':[], 'predictions':[], 'scores':[]}
    test_results={}
   
    for i in range(folds):
        with open('/media/james/ext4data1/current/projects/ramasubbu/innercvfeatures/fold_'+str(i)+'_train.pickle','rb') as f: X_train=pickle.load(f)
        with open('/media/james/ext4data1/current/projects/ramasubbu/innercvfeatures/fold_'+str(i)+'_test.pickle','rb') as f: X_test=pickle.load(f)
                
        fold=[str(i+1)]*len(cv['X_train'][i])
        subjects= cv['X_train'][i]
        labels= cv['y_train'][i]
        
        print('\nFold {}/{}\n'.format(i+1, folds))   
        
        y_train= cv['y_train'][i]
        y_test= cv['y_test'][i]
        
        for j,k in zip(est.keys(), est.values()):
            k.fit(X_train, y_train)
            estimator=[j]*len(subjects)
            predictions= k.predict(X_train)
            scores= [1 if x==y else 0 for x,y in zip(labels, predictions)]
            print(scores)
            train_results['fold'].extend(fold)
            train_results['estimator'].extend(estimator)
            train_results['subjects'].extend(subjects)
            train_results['labels'].extend(labels)
            train_results['predictions'].extend(predictions)
            train_results['scores'].extend(scores)
                          
        
    #print('Test accuracies:\n{}'.format(np.add.reduceat(test_results['test'], np.arange(0, folds, 2))*100/2))    

    with open('/media/james/ext4data1/current/projects/ramasubbu/train_results.pickle', 'wb') as d: pickle.dump(train_results, d, pickle.HIGHEST_PROTOCOL) 
    with open('/media/james/ext4data1/current/projects/ramasubbu/test_results.pickle', 'wb') as d: pickle.dump(test_results, d, pickle.HIGHEST_PROTOCOL) 

    return
    
def OuterFolds():
    with open('/media/james/ext4data1/current/projects/ramasubbu/outer_cv.pickle','rb') as f: cv=pickle.load(f)       
    
    scores={'fold': [], 'train':[], 'test':[]}
    
    for i in range(5):
        with open('/media/james/ext4data1/current/projects/ramasubbu/outercvfeatures/fold_'+str(i)+'_train.pickle','rb') as f:    
            X_train=pickle.load(f)
        with open('/media/james/ext4data1/current/projects/ramasubbu/outercvfeatures/fold_'+str(i)+'_test.pickle','rb') as f:
            X_test=pickle.load(f)
        
        y_train= cv['y_train'][i]
        y_test= cv['y_test'][i]
        
        #est= ensemble.RandomForestClassifier()
        #est= ensemble.ExtraTreesClassifier()
        #est= neighbors.KNeighborsClassifier()
        #est= naive_bayes.GaussianNB()
        #est= tree.DecisionTreeClassifier()
        #est= svm.LinearSVC()
        est= discriminant_analysis.LinearDiscriminantAnalysis()
        
        est.fit(X_train, y_train)
                
        print('\nFold {}/5\n'.format((i+1)))
        
        predicted_train= est.predict(X_train)
        train_score= est.score(X_train, y_train)
        print('X_train predictions: {}'.format(predicted_train))
        print('y_train actual vals: {}'.format(y_train))        
        print('Training set score: {}%\n'.format((train_score*100)))
        
        predicted_test= est.predict(X_test)
        test_score= est.score(X_test, y_test)
        print('X_test predictions: {}'.format(predicted_test))
        print('y_test actual vals: {}'.format(y_test))
        print('Baseline accuracy: {}'.format('50%'))
        print('Test set score: {}%\n'.format((test_score*100)))

        scores['fold'].append(i)
        scores['train'].append(train_score)
        scores['test'].append(test_score)
                
    print('All Train Average Acc: {}%'.format((sum(scores['train'])/5)*100))
    print('All Test Average Acc: {}%'.format((sum(scores['test'])/5)*100))
    #We set this to 50% because of the stratified k-fold CV
    print('All Test Expected Acc: {}%'.format('50%'))
    print('Combined acc per outer fold:\n{}'.format(np.add.reduceat(scores['test'], np.arange(0, 5, 1))*100))    
        
    with open('/media/james/ext4data1/current/projects/ramasubbu/est_scores.pickle', 'wb') as d: pickle.dump(scores, d, pickle.HIGHEST_PROTOCOL) 

    return
