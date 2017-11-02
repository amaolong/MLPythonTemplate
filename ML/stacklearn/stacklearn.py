'''
idea

import all library model api call interface and embedding them into a dictionary that is with keys uniquely defined across the entire system

'''
from param_handling import *
from classification.sklearn_params import *
from classification.xgb_params_sklearn_interface import *
# scikit-learn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as EFC
from sklearn.neighbors import KNeighborsClassifier as KNN
# xgboost
import xgboost as xgb
# lgbm
import lightgbm as lgbm

model_dict={}
model_dict['sksvc']=SVC()
model_dict['skrf']=RFC()
model_dict['skef']=EFC()
model_dict['sknn']=KNN()
model_dict['xgb']=xgb.XGBClassifier()
model_dict['lgbm']=lgbm.LGBMClassifier()


''' handling model parameters '''

'''classification'''










'''regression'''