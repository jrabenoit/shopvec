#!/usr/bin/env python3

import csv, shutil, os, glob, random
import numpy as np

source= '/home/james/Desktop/PAC Data/pac_2016_data_files'

dest= '/home/james/Desktop/PAC Data/pac_2016_data_holdout'

hcfiles= [glob.glob(source + '/*Site_01-HC.mat'),
          glob.glob(source + '/*Site_02-HC.mat'),
          glob.glob(source + '/*Site_03-HC.mat'),
          glob.glob(source + '/*Site_04-HC.mat'),
          glob.glob(source + '/*Site_05-HC.mat'),
          glob.glob(source + '/*Site_06-HC.mat'),
          glob.glob(source + '/*Site_07-HC.mat'),
          glob.glob(source + '/*Site_08-HC.mat'),
          glob.glob(source + '/*Site_09-HC.mat'),
          glob.glob(source + '/*Site_10-HC.mat'),
          glob.glob(source + '/*Site_11-HC.mat')]

mddfiles= [glob.glob(source + '/*Site_01-MDD.mat'),
          glob.glob(source + '/*Site_02-MDD.mat'),
          glob.glob(source + '/*Site_03-MDD.mat'),
          glob.glob(source + '/*Site_04-MDD.mat'),
          glob.glob(source + '/*Site_05-MDD.mat'),
          glob.glob(source + '/*Site_06-MDD.mat'),
          glob.glob(source + '/*Site_07-MDD.mat'),
          glob.glob(source + '/*Site_08-MDD.mat'),
          glob.glob(source + '/*Site_09-MDD.mat'),
          glob.glob(source + '/*Site_10-MDD.mat'),
          glob.glob(source + '/*Site_11-MDD.mat')]

hc_n=[]
mdd_n=[]
for i in range(11):
    hc_site_n= int(round((len(hcfiles[i])/436)*100,0))
    hc_n.append(hc_site_n)
    mdd_site_n= int(round((len(mddfiles[i])/436)*100,0))
    mdd_n.append(mdd_site_n)

holdout_files = []
for i in range(11):
    random.shuffle(hcfiles[i])
    holdout_files.extend(hcfiles[i][:hc_n[i]])
    random.shuffle(mddfiles[i])
    holdout_files.extend(mddfiles[i][:mdd_n[i]])

for file in holdout_files:
    shutil.move(file, dest)
