# imports
import numpy as np
import pandas as pd
import sklearn as sk
import xgboost as xgb
#import lightgbm as lgbm # gcc issue
from sklearn.ensemble import RandomForestClassifier
#from forest import *  # modified random forest with gwava study to deal with class balance
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
import pylab as pl
import sys,os
import gc
import random
import matplotlib
import matplotlib.pyplot as plt
matplotlib.pyplot.switch_backend('agg')


used_class_weight='balanced'
# modified sklearn rf
forest_params={}
forest_params['n_estimators'] = 100
forest_params['max_depth']=None
forest_params['min_samples_split']=2
forest_params['balance_classes']=0
forest_params['compute_importances']=True
forest_params['oob_score']=True
forest_params['n_jobs'] = 5
# sklearn_rf
skrf_params={}
skrf_params['n_estimators'] = 100
skrf_params['criterion'] = 'gini'   # 'entropy'
skrf_params['max_depth'] = 5
skrf_params['max_features'] = 'auto'
skrf_params['min_samples_split'] = 2
skrf_params['oob_score'] = True
skrf_params['class_weight']=used_class_weight
skrf_params['bootstrap'] = True
skrf_params['n_jobs'] = 5
# lgbm
lgbm_params = {}
lgbm_params['max_bin'] = 10
lgbm_params['learning_rate'] = 0.002  # shrinkage_rate
lgbm_params['boosting_type'] = 'gbdt'
# lgbm_params['boosting_type'] = 'rf'
lgbm_params['objective'] = 'binary'
lgbm_params['metric'] = 'mae'  # or 'mae'
# lgbm_params['metric'] = 'binary_logloss'          # or 'mae'
lgbm_params['feature_fraction'] = 1  # feature bagging pct
lgbm_params['bagging_freq'] = 40  # sample bagging per # of iteration
lgbm_params['bagging_fraction'] = 0.85  # sample bagging pct
lgbm_params['num_leaves'] = 512  # num_leaf
lgbm_params['min_data'] = 5  # min_data_in_leaf, check with gwava settings
lgbm_params['min_hessian'] = 0.05  # min_sum_hessian_in_leaf
lgbm_params['verbose'] = 0
lgbm_params['feature_fraction_seed'] = 2
lgbm_params['bagging_seed'] = 3
lgbm_params['max_depth'] = 100
lgbm_params['weight_column'] = ''
# xgboost
xgb_params={}
# objective and evaluation
xgb_params['objective'] = 'binary:logistic'
'''
objective [default=reg:linear]
"reg:linear" -linear regression
"reg:logistic" -logistic regression
"binary:logistic" -logistic regression for binary classification, output probability
"binary:logitraw" -logistic regression for binary classification, output score before logistic transformation
"count:poisson" -poisson regression for count data, output mean of poisson distribution
max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization)
"multi:softmax" -set XGBoost to do multiclass classification using the softmax objective, you also need to set num_class(number of classes)
"multi:softprob" -same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata, nclass matrix. The result contains predicted probability of each data point belonging to each class.
"rank:pairwise" -set XGBoost to do ranking task by minimizing the pairwise loss
"reg:gamma" -gamma regression with log-link. Output is a mean of gamma distribution. It might be useful, e.g., for modeling insurance claims severity, or for any outcome that might be gamma-distributed
"reg:tweedie" -Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any outcome that might be Tweedie-distributed.
'''
xgb_params['eval'] = 'mae'
'''
eval_metric [default according to objective]
evaluation metrics for validation data, a default metric will be assigned according to objective (rmse for regression, and error for classification, mean average precision for ranking )
User can add multiple evaluation metrics, for python user, remember to pass the metrics in as list of parameters pairs instead of map, so that latter "eval_metric" won"t override previous one
The choices are listed below:
"rmse": root mean square error
"mae": mean absolute error
"logloss": negative log-likelihood
"error": Binary classification error rate. It is calculated as #(wrong cases)/#(all cases). For the predictions, the evaluation will regard the instances with prediction value larger than 0.5 as positive instances, and the others as negative instances.
"error@t": a different than 0.5 binary classification threshold value could be specified by providing a numerical value through "t".
"merror": Multiclass classification error rate. It is calculated as #(wrong cases)/#(all cases).
"mlogloss": Multiclass logloss
"auc": Area under the curve for ranking evaluation.
"ndcg":Normalized Discounted Cumulative Gain
"map":Mean average precision
"ndcg@n","map@n": n can be assigned as an integer to cut off the top positions in the lists for evaluation.
"ndcg-","map-","ndcg@n-","map@n-": In XGBoost, NDCG and MAP will evaluate the score of a list without any positive samples as 1. By adding "-" in the evaluation metric XGBoost will evaluate these score as 0 to be consistent under some conditions. training repeatedly
"poisson-nloglik": negative log-likelihood for Poisson regression
"gamma-nloglik": negative log-likelihood for gamma regression
"gamma-deviance": residual deviance for gamma regression
"tweedie-nloglik": negative log-likelihood for Tweedie regression (at a specified value of the tweedie_variance_power parameter)
'''
# training parameters
xgb_params['learning_rate']=0.3   # aka eta:  After each boosting step, we can directly get the weights of new features.
                        # and eta actually shrinks the feature weights to make the boosting process more conservative
xgb_params['max_depth'] = 5
xgb_params['n_estimators']=20
xgb_params['booster']='gbtree'  # gbtree/gblinear/dart
xgb_params['n_jobs']=5
xgb_params['reg_alpha'] = 0.4  # l1 regularization
xgb_params['reg_lambda'] = 0.8  # l2 regularization
xgb_params['gamma'] =0  # default=0, alias: min_split_loss
xgb_params['tree_method']='auto'
'''
Choices: {"auto", "exact", "approx", "hist", "gpu_exact", "gpu_hist"}
"auto": Use heuristic to choose faster one.
For small to medium dataset, exact greedy will be used.
For very large-dataset, approximate algorithm will be chosen.
Because old behavior is always use exact greedy in single machine, user will get a message when approximate algorithm is chosen to notify this choice.
"exact": Exact greedy algorithm.
"approx": Approximate greedy algorithm using sketching and histogram.
"hist": Fast histogram optimized approximate greedy algorithm. It uses some performance improvements such as bins caching.
"gpu_exact": GPU implementation of exact algorithm.
"gpu_hist": GPU implementation of hist algorithm.
'''
xgb_params['subsample'] = 0.8
xgb_params['colsample_bytree'] = 1
xgb_params['colsample_bylevel'] = 1
xgb_params['scale_pos_weight'] = max(matching_type[_matching].label.value_counts().values) \
                                 / min(matching_type[_matching].label.value_counts().values)
xgb_params['base_score']=0.5    # base score for all predictions
xgb_params['sketch_eps'] = 0.03  # default, used in approximation algorithms
xgb_params['silent']=0          # 0 means printing running messages, 1 means silent mode.

xgb_params['grow_policy']= 'depthwise'   # 'lossguide'

# xgb_params['updater']
'''
A comma separated string defining the sequence of tree updaters to run, providing a modular way to construct and to modify the trees. This is an advanced parameter that is usually set automatically, depending on some other parameters. However, it could be also set explicitely by a user. The following updater plugins exist:
"grow_colmaker": non-distributed column-based construction of trees.
"distcol": distributed tree construction with column-based data splitting mode.
"grow_histmaker": distributed tree construction with row-based data splitting based on global proposal of histogram counting.
"grow_local_histmaker": based on local histogram counting.
"grow_skmaker": uses the approximate sketching algorithm.
"sync": synchronizes trees in all distributed nodes.
"refresh": refreshes tree"s statistics and/or leaf values based on the current data. Note that no random subsampling of data rows is performed.
"prune": prunes the splits where loss < min_split_loss (or gamma).
In a distributed setting, the implicit updater sequence value would be adjusted as follows:
"grow_histmaker,prune" when dsplit="row" (or default) and prob_buffer_row == 1 (or default); or when data has multiple sparse pages
"grow_histmaker,refresh,prune" when dsplit="row" and prob_buffer_row < 1
"distcol" when dsplit="col"
'''
#
model_params={
    'forest':forest_params,
    'skrf':skrf_params,
    'xgb':xgb_params,
    'lgbm':lgbm_params
}
## set parameters
if method == 'skrf':
    clf = sk.ensemble.RandomForestClassifier(**method_param_dict[method])
if method == 'xgb':
    pass
## set prediction
if method == 'forest' or method == 'skrf':
    print('current_cv_', i, ' ', 'train size: ', train.shape, ' ', 'test size: ', test.shape, '\t',
          'test pos label #: ', np.sum(y[test]))
    probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
    pred = probas_[:, 1]
if method == 'xgb':
    xgb_data = xgb.DMatrix(X[train], label=y[train])
    xgb_data_test = xgb.DMatrix(X[test])
    clf = xgb.train(method_param_dict[method], xgb_data, num_boost_round=20)
    pred = clf.predict(xgb_data_test)
## set K fold stratified cross validation
for i, (train, test) in enumerate(cv.split(df.drop(cls, axis=1).values,df[cls].values)):
    pass

## record roc curve

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
    for _ in range(10): ## cross validation
        fpr, tpr, thresholds = curve_type(y[test], pred)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        if do_plot:
            pl.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))  # cross validation roc curve

if do_plot:
    pl.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Chance')
mean_tpr /= k
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
# plotting
if do_plot:
    pl.plot(mean_fpr, mean_tpr, 'k--',
            label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)
    pl.xlim([-0.05, 1.05])
    pl.ylim([-0.05, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title('10-fold cross validation')
    pl.legend(loc="lower right")
    figure_name = outdir + method + '_' + matching + '_10-fold_cv.pdf'
    pl.savefig(figure_name)
    pl.close()
## feature importance
def feat_imp(model, df, method, cls='label'):
    '''
    :param model:
    :param df:
    :param cls:
    :return:
    '''
    if method=='forest' or method=='skrf':
        idx = model.feature_importances_.argsort()
        fis = pd.Series(model.feature_importances_[idx[::-1]], index=df.columns.drop(cls)[idx[::-1]].values)
    if method=='xgb':
        feat_imp_data=np.zeros(df.columns.drop(cls).shape)
        tmp=pd.DataFrame.from_dict(model.get_fscore(),orient='index').reset_index()
        tmp['idx']=tmp['index']
        tmp_vec=tmp['index'].values
        tmp['idx'] = list(map(lambda x: int(x.replace('f', '')), tmp_vec))
        for i, _ in enumerate(tmp['idx']):
            feat_imp_data[_]=tmp[0][i]
        fis = pd.Series(feat_imp_data/feat_imp_data.sum(), index=df.columns.drop(cls).values)
    fis.sort_values(ascending=False,inplace=True)
    return fis