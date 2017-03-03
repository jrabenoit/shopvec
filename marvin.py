#!/usr/bin/env python3

import pprint, itertools, gc, sys, pickle
import inputs,crossval,iterator#,comparator,bootstrap,visualize
from collections import defaultdict

#Select a group of scans to use 
print('Step 1: Create reference dict with scans set up for ML')
inputs.SubjectInfo()
inputs.MassageScans()

print('Step 2: Create outer & inner cross-validation sets')
crossval.OuterCv()
crossval.InnerCv()

'''
print('Step 3: Try all feature manipulation & ML estimator combinations')
iterator.TryMLTools()
'''
#print('Step 8: Pick Best featsel/decomp/mltool Combo')
#fold_index, folds = comparator.PickBest(test_results)

#print('Step 9: Run Best Combo on Outer CV Holdout')
#final_train_results, final_test_results, final_train_predictions, final_test_predictions, final_train_labels, final_test_labels = iterator.TestHoldout(oX_train, oX_test, oy_train, oy_test, fold_index, iter_n) 

#print('Step 10: Print Test vs. Chance Results')
#final_train_correct, final_test_correct = comparator.PrintFinal(final_train_results, final_test_results, n_1, n_2, final_train_predictions, final_test_predictions, final_train_labels, final_test_labels, iter_n, train_index_files, test_index_files)

#Build accuracy profile for each subject
#subject_results_test, subject_results_train = comparator.SubjectAccuracy(iter_n, final_train_correct, final_test_correct, train_index_files, test_index_files, concat_subjects_dict)

#Concatenate classification attempts for each subject
#for key, value in subject_results_test.items():
#    concatenated_test[key].append(value)

#for key, value in subject_results_train.items():
#    concatenated_train[key].append(value)

#print('>>>CHAINING TRAINING RESULTS TOGETHER')
#concatenated_train_chained = defaultdict(list)
#for key, value in concatenated_train.items():
#    concatenated_train_chained[key] = list(itertools.chain.from_iterable(value))
    
#print('>>>CHAINING TEST RESULTS TOGETHER')
#concatenated_test_chained = defaultdict(list)
#for key, value in concatenated_test.items():
#    concatenated_test_chained[key] = list(itertools.chain.from_iterable(value))

#print('>>>TRAIN SUBJECT ACCURACY SCORES')
#per_subject_train_acc = defaultdict(list)
#for key, value in concatenated_train_chained.items():
#    per_subject_train_acc[key] = round((sum(value)/len(value))*100,4)
#pprint.pprint(per_subject_train_acc)

#print('>>>TEST SUBJECT ACCURACY SCORES')
#per_subject_test_acc = defaultdict(list)
#for key, value in concatenated_test_chained.items():
#    per_subject_test_acc[key] = round((sum(value)/len(value))*100,4)
#pprint.pprint(per_subject_test_acc)

#final_acc = sum(list(per_subject_test_acc.values()))/len(list(per_subject_test_acc.values()))
#print('\n>>>AVERAGE ACCURACY: {}%'.format(round(final_acc,2)))
#p_value = bootstrap.EmpiricalDistro(n_1, n_2, per_subject_test_acc)
#print('>>>P VALUE UNCORRECTED: {}'.format(p_value))
#print('>>>P VALUE CORRECTED: {}'.format(p_value*3))

#print('>>>SAMPLE SIZE: {}'.format(n_1 + n_2))

