#!/usr/bin/env python3

from collections import defaultdict
import pprint
import itertools
import copy
import operator
import numpy as np
from scipy.stats import chisquare
from pprint import pprint

def PickBest(test_results):
    '''returns key for fold with min variance of the 3 sets of test parameters giving the highest accuracy on the inner loop holdout set'''
    test_results_copy = copy.copy(test_results)

#Get mean for each fold, remove two lowest accuracy items, change accuracy to variance, pick combination of parameters giving minimum variance for each fold.
    fold_1_mean = sum(value for key, value in test_results_copy.items() if 'test_1' in key.lower())/5
    fold_2_mean = sum(value for key, value in test_results_copy.items() if 'test_2' in key.lower())/5
    fold_3_mean = sum(value for key, value in test_results_copy.items() if 'test_3' in key.lower())/5
    fold_4_mean = sum(value for key, value in test_results_copy.items() if 'test_4' in key.lower())/5
    fold_5_mean = sum(value for key, value in test_results_copy.items() if 'test_5' in key.lower())/5
#Get dict elements (param sets) with highest 3 accuracies for each fold
    top_3_accs={}
    top_3_accs.update(dict(sorted(((k, v) for k, v in test_results_copy.items() if 'test_1' in k), key=operator.itemgetter(1), reverse=True)[:3]))
    top_3_accs.update(dict(sorted(((k, v) for k, v in test_results_copy.items() if 'test_2' in k), key=operator.itemgetter(1), reverse=True)[:3]))
    top_3_accs.update(dict(sorted(((k, v) for k, v in test_results_copy.items() if 'test_3' in k), key=operator.itemgetter(1), reverse=True)[:3]))
    top_3_accs.update(dict(sorted(((k, v) for k, v in test_results_copy.items() if 'test_4' in k), key=operator.itemgetter(1), reverse=True)[:3]))
    top_3_accs.update(dict(sorted(((k, v) for k, v in test_results_copy.items() if 'test_5' in k), key=operator.itemgetter(1), reverse=True)[:3]))
#Change accuracy to variance
    top_3_accs.update((k, (v-fold_1_mean)**2) for k,v in top_3_accs.items() if 'test_1' in k)
    top_3_accs.update((k, (v-fold_2_mean)**2) for k,v in top_3_accs.items() if 'test_2' in k)
    top_3_accs.update((k, (v-fold_3_mean)**2) for k,v in top_3_accs.items() if 'test_3' in k)
    top_3_accs.update((k, (v-fold_4_mean)**2) for k,v in top_3_accs.items() if 'test_4' in k)
    top_3_accs.update((k, (v-fold_5_mean)**2) for k,v in top_3_accs.items() if 'test_5' in k)
#Select key with minimum variance from remaining 3 elements per fold with highest accuracy
    folds = {}
    fold_1 = min({k:v for (k,v) in top_3_accs.items() if 'test_1' in k}, key=top_3_accs.get)
    fold_2 = min({k:v for (k,v) in top_3_accs.items() if 'test_2' in k}, key=top_3_accs.get)
    fold_3 = min({k:v for (k,v) in top_3_accs.items() if 'test_3' in k}, key=top_3_accs.get)
    fold_4 = min({k:v for (k,v) in top_3_accs.items() if 'test_4' in k}, key=top_3_accs.get)
    fold_5 = min({k:v for (k,v) in top_3_accs.items() if 'test_5' in k}, key=top_3_accs.get)
    
    fold_index= [fold_1, fold_2, fold_3, fold_4, fold_5]

    folds[fold_1] = top_3_accs[fold_1]
    folds[fold_2] = top_3_accs[fold_2]
    folds[fold_3] = top_3_accs[fold_3]
    folds[fold_4] = top_3_accs[fold_4]
    folds[fold_5] = top_3_accs[fold_5]    
    
    pprint(folds)
    return fold_index, folds

################################################################################

def PrintFinal(final_train_results, final_test_results, n_1, n_2, final_train_predictions, final_test_predictions, final_train_labels, final_test_labels, iter_n, train_index_files, test_index_files):   

#    pprint(final_train_results)    
    final_average_train = round(sum(final_train_results.values())/5, 3)
#    print('\n>>> Average of best tool across training sets: {}% <<<\n'.format(round(final_average_train, 2)))

#    pprint(final_test_results)
    final_average_test = round(sum(final_test_results.values())/5, 3)
#    print('\n>>> Average Tested Accuracy: {}% <<<\n'.format(round(final_average_test, 3)))
    
    n_max = max(n_1, n_2)
    n_min = min(n_1, n_2)
    expected_accuracy = (n_max/(n_min + n_max))*100
#    print('>>> Chance Accuracy For n_1= {} and n_2= {}: {}% <<<\n'.format(n_1, n_2, round(expected_accuracy, 3)))
    acc_diff = final_average_test - expected_accuracy
    if acc_diff > 0:
        acc_direction = str('above')
    elif acc_diff < 0:
        acc_direction = str('below')
    else: 
        acc_direction = str('equals')
#    print('>>> Tested accuracy {} chance accuracy by {}% <<<\n'.format(acc_direction, round(abs(acc_diff), 3)))
    
# Create results dict
    final_test_correct = defaultdict(list)
    final_train_correct = defaultdict(list)
    for i in range(iter_n):
        for j in range(5):
            list_final_test_correct= []
            list_final_train_correct= []
            for k in range(len(final_test_predictions[i][j])):
                if final_test_predictions[i][j][k]==final_test_labels[i][j][k]:
                    list_final_test_correct.append(1)
                else:
                    list_final_test_correct.append(0)
            for k in range(len(final_train_predictions[i][j])):
                if final_train_predictions[i][j][k]==final_train_labels[i][j][k]:
                    list_final_train_correct.append(1)
                else:
                    list_final_train_correct.append(0)
            final_test_correct[i] += [list_final_test_correct]  
            final_train_correct[i] += [list_final_train_correct]  
    return final_train_correct, final_test_correct

#Removed chi-square test; inappropriate for new stats
'''
    #This is a one-way chi square test
    #We are using chi square because of testing a single categorical variable for testing whether our sample data distribution is consistent with a theoretical data distribution (is it generalizable?)
    chi_square_test = chisquare(count_obs_test, f_exp= count_exp_test, ddof=0)
    print('>>> (chi square, raw p-value): {}\n <<<'.format(str(chi_square_test)))
'''

def SubjectAccuracy(iter_n, final_train_correct, final_test_correct, train_index_files, test_index_files, concat_subjects_dict):
    
    list_of_test_results= []
    list_of_test_subjects= []
    for i in range(iter_n):
        for j in range(5):
            for k in range(len(final_test_correct[i][j])):
                list_of_test_results.append(final_test_correct[i][j][k])
                list_of_test_subjects.append(test_index_files[i][j][k])
    
    list_of_train_results= []
    list_of_train_subjects= []
    for i in range(iter_n):
        for j in range(5):
            for k in range(len(final_train_correct[i][j])):
                list_of_train_results.append(final_train_correct[i][j][k])
                list_of_train_subjects.append(train_index_files[i][j][k])
    
    subject_results_test= defaultdict(list)
    for subj,result in zip(list_of_test_subjects,list_of_test_results):
        if subj in subject_results_test:
            subject_results_test[subj] += [result]
        else:
            subject_results_test[subj] = [result]

    subject_results_train= defaultdict(list)   
    for subj,result in zip(list_of_train_subjects,list_of_train_results):
        if subj in subject_results_train:
            subject_results_train[subj] += [result]
        else:
            subject_results_train[subj] = [result]
    return subject_results_test, subject_results_train
    


