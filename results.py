#!/usr/bin/env python3 

import pickle
import numpy as np
import pandas as pd

def Scores():
    with open('/media/james/ext4data1/current/projects/ramasubbu/outer_test_results_group_'+str(group)+'_run_'+str(run)+'.pickle', 'rb') as f: =pickle.load(f)
