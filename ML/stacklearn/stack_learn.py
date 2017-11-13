'''
idea

import all library model api call interface and embedding them into a dictionary that
is with keys uniquely defined across ntire system

'''
# customized class and functions
from ML.stacklearn.param_handling import *
from ML.stacklearn.classification.sklearn_params import *
from ML.stacklearn.classification.xgb_params_sklearn_interface import *
from ML.stacklearn.stack_model import *
# scikit-learn
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import ExtraTreesClassifier as EFC
from sklearn.neighbors import KNeighborsClassifier as KNN
# xgboost
import xgboost as xgb
# lgbm
import lightgbm as lgbm
# general import
import numpy as np
#
model_dict={}
model_dict['sksvc']=SVC
model_dict['skrf']=RFC
model_dict['skef']=EFC
model_dict['sknn']=KNN
model_dict['xgb']=xgb.XGBClassifier
model_dict['lgbm']=lgbm.LGBMClassifier

''' FLAGS '''
model_level=2    # 2 or 3
debug=True


''' handling model parameters '''
# pull model type, populate different parameters and sample a few to be used as base learners

models=populate_params(param_collection_sk_default,param_collection_names_sk_default,0)
for _ in populate_params(param_collection_xgb_lgbm_default,param_collection_names_xgb_lgbm_default,0):
    models.append(_)

for _ in models:
    _.sample()

model_count=0
for _ in models:
    for _2 in _.sampled_model_params:
        model_count+=1
print('total models sampled: ',model_count)


'''insert model params'''
level_1_models=[]   # a list
level_2_models=[]   # 1 or a list
level_3_models=[]   # optional

# insert lv1 model
for _ in models:
    for _2 in _.sampled_model_params:
        t_model=model_dict[_.model_name].set_params(**_2)
        level_1_models.append(t_model)

# insert lv2 model
for _ in models:
    for _2 in _.sampled_model_params:
        t_model = model_dict[_.model_name].set_params(**_2)
        level_2_models.append(t_model)

# insert lv3 model
for _ in models:
    for _2 in _.sampled_model_params:
        t_model = model_dict[_.model_name].set_params(**_2)
        level_3_models.append(t_model)

# visual check
for _ in level_1_models:
    print(_)

for _ in level_2_models:
    print(_)

for _ in level_3_models:
    print(_)


X=np.random.random([50,10])   # 50 obs each with 10 features
y=np.random.ranint(0,2,50)    # 50 labels

X1=np.random.random([50,10])   # 50 obs each with 10 features
y1=np.random.ranint(0,2,50)    # 50 labels

'''prepare model data X, y'''

# call stack_model
clf=stack_model(X,y,level_1_models,level_2_models)

clf.fit_2layers()
pred1=clf.predict_2layers(X1)

# check accuracy between pred1 and y1

# 10 fold implementation







'''classification'''


