'''
    general import stuff
'''

import xlrd # read spreadsheet
import sys
import os   # cmd call os.system('cmd')
import argparse
import csv
from random import randint
import pickle
import numpy as np
import scipy as sc
'''
    visual output
    https://matplotlib.org/2.0.2/api/index.html
    http://seaborn.pydata.org/api.html

'''
import matplotlib.pyplot as plt
# %matplotlib inline   # uncomment for jupyter notebook
import seaborn as sns
color = sns.color_palette()
import pylab as pl  # for basic plotting
# plt.switch_backend('agg')  # uncomment for plotting in linux
'''
    data frame
    https://pandas.pydata.org/pandas-docs/stable/api.html
'''
import pandas as pd
'''
    python version of command line
    https://docs.python.org/2/library/subprocess.html
'''
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
from subprocess import call
'''
cmd = 'command line operations'
call(cmd, shell=True)
'''



''' 
    skicit learn ML package 
    http://scikit-learn.org/stable/modules/classes.html
'''
# classification
from sklearn.svm import SVC # use different kernels in base learner
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import BaggingClassifier as BC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.linear_model import LogisticRegression as LogR
from sklearn.neighbors import KNeighborsClassifier as KNN
# regression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import AdaBoostRegressor as ADR
from sklearn.ensemble import BaggingRegressor as BR
from sklearn.linear_model import Lasso as LS
from sklearn.linear_model import Ridge as RG
from sklearn.linear_model import LinearRegression as LR
# model selection
from sklearn.model_selection import KFold as KF
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.model_selection import cross_val_score as CVS  # SKF in
from sklearn.model_selection import GridSearchCV as GSCV
# feature selection
from sklearn.feature_selection import RFECV
# metric
from sklearn.metrics import *




''' 
    deep neural networks & tensorflow 
    https://www.tensorflow.org/api_docs/
    https://keras.io/getting-started/functional-api-guide/
'''
import tensorflow as tf   # tensorflow only for python >= 3.5
import keras as ks
import h2o

# look into APIs in tensorflow.learn.*





