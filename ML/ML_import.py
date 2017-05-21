''' general import stuff '''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function   # print function in python 3.x
import xlrd # read spreadsheet
import numpy as np
import sys
import os
import argparse
import csv
from random import randint
import pickle

''' skicit learn ML package '''
from sklearn.svm import SVC
from sklearn.svm import SVR

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import BaggingRegressor as BR
from sklearn.ensemble import AdaBoostClassifier

from sklearn.linear_model import LogisticRegression as LogR
from sklearn.linear_model import Lasso as LS
from sklearn.linear_model import Ridge as RG
from sklearn.linear_model import LinearRegression as LR

from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.model_selection import cross_val_score as CVS
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import KFold as KF
from sklearn.model_selection import GridSearchCV as GSCV

from sklearn.feature_selection import RFECV
from sklearn.metrics import *

''' tensorflow 
import tensorflow as tf   # tensorflow only for python >= 3.5

# look into APIs in tensorflow.learn.*

'''
