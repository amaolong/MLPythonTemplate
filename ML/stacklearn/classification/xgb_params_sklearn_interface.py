''' xgboost and lightgbm for classification'''

'''
xgboost parameters through scikit-learn interface

XGBClassifier(base_score=0.5,
colsample_bylevel=1,
colsample_bytree=1,
gamma=0,
learning_rate=0.1,
max_delta_step=0,
max_depth=3,
min_child_weight=1,
missing=None,
n_estimators=100,
nthread=-1,
objective='binary:logistic',
reg_alpha=0,
reg_lambda=1,
scale_pos_weight=1,
subsample=1)
'''
# 24 total combinations
xgb_params={}
# training parameters
xgb_params['base_score']=[0.5]
xgb_params['booster']=['gbtree','dart']  # gbtree/gblinear/dart
xgb_params['learning_rate']=[0.1]
xgb_params['max_depth'] = [5,10]
xgb_params['n_estimators']=[50,100,200]
xgb_params['subsample'] = [0.8,1]
xgb_params['objective'] =['binary:logistic']
xgb_params['n_jobs']=[5]
xgb_params['gamma'] =[0]  # default=0, alias: min_split_loss
xgb_params['reg_alpha'] = [0.4]  # l1 regularization
xgb_params['reg_lambda'] = [0.8]  # l2 regularization
xgb_params['colsample_bytree'] = [1]
xgb_params['colsample_bylevel'] = [1]


'''
lightgbm parameters through scikit-learn interface

LGBMClassifier(boosting_type='gbdt', 
colsample_bytree=1, 
learning_rate=0.1,
max_bin=255, 
max_depth=-1, 
min_child_samples=10,
min_child_weight=5, 
min_split_gain=0, 
n_estimators=10, 
nthread=-1,
num_leaves=31, 
objective='binary', 
reg_alpha=0, 
reg_lambda=0,
seed=0, 
silent=True, 
subsample=1, 
subsample_for_bin=50000,
subsample_freq=1)
'''
# 24 total combinations
lgbm_params={}
lgbm_params['boosting_type']=['gbdt','dart']
'''
gbdt, traditional Gradient Boosting Decision Tree. 
dart, Dropouts meet Multiple Additive Regression Trees. 
goss, Gradient-based One-Side Sampling. 
rf, Random Forest.
'''
lgbm_params['num_leaves'] =[31]
lgbm_params['max_depth'] =[5,10]
lgbm_params['learning_rate']=[0.1]
lgbm_params['n_estimators'] =[50,100,200]
lgbm_params['subsample_for_bin'] =[50000]
lgbm_params['subsample'] =[0.8,1]
lgbm_params['subsample_freq'] =[1]
lgbm_params['objective'] =['binary']
lgbm_params['colsample_bytree'] =[1]
lgbm_params['max_bin']=[255]
lgbm_params['min_child_samples']=[20]
lgbm_params['min_child_weight'] =[0.001]
lgbm_params['min_split_gain'] =[0]
lgbm_params['nthread'] =[5]
lgbm_params['reg_alpha'] =[0.4]
lgbm_params['reg_lambda'] =[0.8]


param_collection_xgb_lgbm_default=[xgb_params,lgbm_params]
param_collection_names_xgb_lgbm_default=['xgb','lgbm']