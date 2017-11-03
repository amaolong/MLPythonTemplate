'''
idea

import all library model api call interface and embedding them into a dictionary that
is with keys uniquely defined across ntire system

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
# pull model type, populate different parameters and sample a few to be used as base learners

models=populate_params(param_collection_names_sk_default,param_collection_names_sk_default,0)
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
level_1_models=[]
for _ in models:
    for _2 in _.sampled_model_params:
        t_model=model_dict[_.model_name].set_params(**_2)
        level_1_models.append(t_model)
# visual check
for _ in level_1_models:
    print(_)

# call stack_model



'''prepare model data X, y'''


# 10 fold implementation
'''classification'''


