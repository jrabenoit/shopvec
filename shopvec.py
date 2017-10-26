#!/usr/bin/env python3

import os
import crossval, features, estimators, estpicker, bootstrap

def Vecuum():
    
    print('\nGROUP CHOICES (automated order: 2,1,0,4,3,5,11,10,12)')
    print('\nSymptom Severity:\n')
    print(' 0= control/mild\n 1= control/severe\n 2= control/very severe\n 3= mild/severe\n 4= mild/very severe\n 5= severe/very severe\n 6= control/all patients\n')
    
    print('Treatment Response:\n')
    print(' 7= control/non-responder\n 8= control/all responder\n 9= control/remitter only\n 10= non-responder/all responder\n 11= non-responder/remitter only\n 12= responder vs remitter\n 13= control/all patients\n')

    groups= [2,1,0,
            4,3,5,
            11,10,12]
    runs=5

    for group in groups:
        run=1
        for i in range(runs):    
            print('BEGINNING RUN {}/{}'.format(run, runs))
            crossval.OuterCV(group)
            crossval.InnerCV()    
            features.SelKBest()
            features.SelKBestOuter()
            estimators.InnerFolds(group, run)
            bestest= estpicker.Best(group, run)
            estimators.OuterFolds(group, run, bestest)
            bootstrap.Bill(group, run)
            run= run + 1
            print('RUN COMPLETE')    

    os.system('spd-say -r -50 -p -50 -t female3 "your groups have finished running. To run more groups, you must construct additional pylons."')
    
    return

