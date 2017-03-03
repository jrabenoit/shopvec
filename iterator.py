#!/usr/bin/env python3

import numpy as np
import pprint, itertools, copy, re, inspect
import features, transform, estimators, results
import pandas as pd
from collections import defaultdict

def TryMLTools():    
    features_set= inspect.getmembers(features, inspect.isfunction)
    transform_set= inspect.getmembers(transform, inspect.isfunction)
    estimators_set= inspect.getmembers(estimators, inspect.isfunction)
    
    try_sets= list(itertools.product(features_set, transform_set, estimators_set))
    
    for i in range(len(try_sets)):
        this_set= str(sets[i][0][0]+ ', '+ sets[i][1][0]+ ', '+ sets[i][2][0])
        print('Set: {}'.format(seti))
        inner_cv= sets[i][0][1]()
        inner_cv= sets[i][1][1]()
        inner_cv= sets[i][2][1]()
        inner_cv= results.InnerAverages(inner_cv, seti)

    return   

'''
        results.InnerAverages(df_inner_cv)
        train_results.update(ir_train)
        test_results.update(ir_test)  
    pprint.pprint(train_results)
    pprint.pprint(test_results)
    return test_results, param_set_list

def TestHoldout(oX_train, oX_test, oy_train, oy_test, fold_index, iter_n):
    hX_train= copy.copy(oX_train)
    hX_test= copy.copy(oX_test)
    hy_train= copy.copy(oy_train)
    hy_test= copy.copy(oy_test)

    params_fold_1 = list(re.compile('\w+').findall(fold_index[0]))
    params_fold_2 = list(re.compile('\w+').findall(fold_index[1]))
    params_fold_3 = list(re.compile('\w+').findall(fold_index[2]))
    params_fold_4 = list(re.compile('\w+').findall(fold_index[3]))
    params_fold_5 = list(re.compile('\w+').findall(fold_index[4]))
    
    final_params = [params_fold_1, params_fold_2, params_fold_3, params_fold_4, params_fold_5]

    final_train_results= defaultdict(list)
    final_test_results= defaultdict(list)
    #Using list of lists so I know that the labels remain ordered
    final_train_predictions= defaultdict(list)
    final_test_predictions= defaultdict(list)
    final_train_labels= defaultdict(list)
    final_test_labels= defaultdict(list)

    for i in range(iter_n):    
        for j in range(5):
            print('  Final params, fold {0}/{1}: {2}'.format(j+1, len(final_params), final_params[j][1:4]))
            param_set = final_params[j]
        #Random Forest Only, no featsel or decomp
            lX_train, lX_test, lX_train_predict, lX_test_predict, ly_train_labels, ly_test_labels = eval("mltools.ml_func_dict_final['{}'](hX_train[i][j], hX_test[i][j], hy_train[i][j], hy_test[i][j])".format(str(param_set[1])))
        
            oR_train, oR_test = results.OuterAverages(lX_train, lX_test, param_set)
            final_train_results.update(oR_train)
            final_test_results.update(oR_test)
            final_train_predictions[i].append(lX_train_predict)
            final_test_predictions[i].append(lX_test_predict)
            final_train_labels[i].append(ly_train_labels)
            final_test_labels[i].append(ly_test_labels)        
    print('oR_train')
    pprint.pprint(oR_train)
    print('oR_test')
    pprint.pprint(oR_test)
    print('final_train_predictions')
    pprint.pprint(final_train_predictions)
    print('final_test_predictions')
    pprint.pprint(final_test_predictions)
    print('final_train_labels')
    pprint.pprint(final_train_labels)
    print('final_test_labels')
    pprint.pprint(final_test_labels)

    return final_train_results, final_test_results, final_train_predictions, final_test_predictions, final_train_labels, final_test_labels
'''       
