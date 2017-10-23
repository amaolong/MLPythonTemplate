''' additional base learner parameters for classification '''

'''lgbm'''
lgbm_params = {}
lgbm_params['max_bin'] = [10]
lgbm_params['learning_rate'] = [0.002]  # shrinkage_rate
lgbm_params['boosting_type'] = ['gbdt']
# lgbm_params['boosting_type'] = ['rf']
lgbm_params['objective'] = ['binary']
lgbm_params['metric'] = ['mae']  # or 'mae'
lgbm_params['feature_fraction'] = [1]  # feature bagging pct
lgbm_params['bagging_freq'] = [40]  # sample bagging per # of iteration
lgbm_params['bagging_fraction'] = [0.85]  # sample bagging pct
lgbm_params['num_leaves'] = [512]  # num_leaf
lgbm_params['min_data'] = [5]  # min_data_in_leaf, check with gwava settings
lgbm_params['min_hessian'] = [0.05]  # min_sum_hessian_in_leaf
lgbm_params['verbose'] = [0]
lgbm_params['feature_fraction_seed'] = [2]
lgbm_params['bagging_seed'] = [3]
lgbm_params['max_depth'] = [100]
lgbm_params['weight_column'] = ''

'''xgboost'''
xgb_params={}
# objective and evaluation
xgb_params['objective'] = ['binary:logistic']
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
xgb_params['eval'] = ['mae']
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
xgb_params['learning_rate']=[0.3]
'''
aka eta:  After each boosting step, we can directly get the weights of new features.
and eta actually shrinks the feature weights to make the boosting process more conservative
'''
xgb_params['max_depth'] = [5]
xgb_params['n_estimators']=[20]
xgb_params['booster']=['gbtree']  # gbtree/gblinear/dart
xgb_params['n_jobs']=[5]
xgb_params['reg_alpha'] = [0.4]  # l1 regularization
xgb_params['reg_lambda'] = [0.8]  # l2 regularization
xgb_params['gamma'] =[0]  # default=0, alias: min_split_loss
xgb_params['tree_method']=['auto']
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
xgb_params['subsample'] = [0.8]
xgb_params['colsample_bytree'] = [1]
xgb_params['colsample_bylevel'] = [1]
#xgb_params['scale_pos_weight'] = max(matching_type[_matching].label.value_counts().values) / min(matching_type[_matching].label.value_counts().values)
xgb_params['base_score']=[0.5 ]   # base score for all predictions
xgb_params['sketch_eps'] = [0.03] # default, used in approximation algorithms
xgb_params['silent']=[0]          # 0 means printing running messages, 1 means silent mode.
xgb_params['grow_policy']= ['depthwise']   # 'lossguide'
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

param_collection=[xgb_params,lgbm_params]
param_collection_names=['xgb_params','lgbm_params']

