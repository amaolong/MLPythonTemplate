'''scikit-learn base learner parameters for classification'''

# skrf : scikit-learn random forest
skrf_params={}
skrf_params['n_estimators'] = [100,500,1000]
skrf_params['criterion'] = ['gini', 'entropy']
skrf_params['max_depth'] = [5,10]
skrf_params['max_features'] = ['auto']
skrf_params['min_samples_split'] = [2]
skrf_params['oob_score'] = [True]
skrf_params['class_weight'] = ['balanced']
skrf_params['bootstrap'] = [True]
skrf_params['n_jobs'] = [5]
'''
n_estimators : integer, optional (default=10)
The number of trees in the forest.
criterion : string, optional (default=”gini”)
The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain. Note: this parameter is tree-specific.
max_features : int, float, string or None, optional (default=”auto”)
The number of features to consider when looking for the best split:
If int, then consider max_features features at each split.
If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split.
If “auto”, then max_features=sqrt(n_features).
If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
If “log2”, then max_features=log2(n_features).
If None, then max_features=n_features.
Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.
max_depth : integer or None, optional (default=None)
The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
min_samples_split : int, float, optional (default=2)
The minimum number of samples required to split an internal node:
If int, then consider min_samples_split as the minimum number.
If float, then min_samples_split is a percentage and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
Changed in version 0.18: Added float values for percentages.
min_samples_leaf : int, float, optional (default=1)
The minimum number of samples required to be at a leaf node:
If int, then consider min_samples_leaf as the minimum number.
If float, then min_samples_leaf is a percentage and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
Changed in version 0.18: Added float values for percentages.
min_weight_fraction_leaf : float, optional (default=0.)
The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
max_leaf_nodes : int or None, optional (default=None)
Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
min_impurity_split : float,
Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.
Deprecated since version 0.19: min_impurity_split has been deprecated in favor of min_impurity_decrease in 0.19 and will be removed in 0.21. Use min_impurity_decrease instead. 
min_impurity_decrease : float, optional (default=0.)
A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
The weighted impurity decrease equation is the following:
N_t / N * (impurity - N_t_R / N_t * right_impurity
                    - N_t_L / N_t * left_impurity)
where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.
N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.
New in version 0.19.
bootstrap : boolean, optional (default=True)
Whether bootstrap samples are used when building trees.
oob_score : bool (default=False)
Whether to use out-of-bag samples to estimate the generalization accuracy.
n_jobs : integer, optional (default=1)
The number of jobs to run in parallel for both fit and predict. If -1, then the number of jobs is set to the number of cores.
random_state : int, RandomState instance or None, optional (default=None)
If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
verbose : int, optional (default=0)
Controls the verbosity of the tree building process.
warm_start : bool, optional (default=False)
When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest.
class_weight : dict, list of dicts, “balanced”,
“balanced_subsample” or None, optional (default=None) Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
Note that for multioutput (including multilabel) weights should be defined for each class of every column in its own dict. For example, for four-class multilabel classification weights should be [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of [{1:1}, {2:5}, {3:1}, {4:1}].
The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
The “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown.
For multi-output, the weights of each column of y will be multiplied.
Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
'''
# sket : scikit-learn extreme tree
sket_params={}
sket_params['n_estimators'] = [100,500,1000]
sket_params['criterion'] = ['gini', 'entropy']
sket_params['max_depth'] = [5,10]
sket_params['max_features'] = ['auto']
sket_params['min_samples_split'] = [2]
sket_params['oob_score'] = [True]
sket_params['class_weight']=['balance']
sket_params['bootstrap'] = [True]
sket_params['n_jobs'] = [5]
'''
criterion : string, optional (default=”gini”)
The function to measure the quality of a split. Supported criteria are “gini” for the Gini impurity and “entropy” for the information gain.
splitter : string, optional (default=”best”)
The strategy used to choose the split at each node. Supported strategies are “best” to choose the best split and “random” to choose the best random split.
max_depth : int or None, optional (default=None)
The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
min_samples_split : int, float, optional (default=2)
The minimum number of samples required to split an internal node:
If int, then consider min_samples_split as the minimum number.
If float, then min_samples_split is a percentage and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.
Changed in version 0.18: Added float values for percentages.
min_samples_leaf : int, float, optional (default=1)
The minimum number of samples required to be at a leaf node:
If int, then consider min_samples_leaf as the minimum number.
If float, then min_samples_leaf is a percentage and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.
Changed in version 0.18: Added float values for percentages.
min_weight_fraction_leaf : float, optional (default=0.)
The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.
max_features : int, float, string or None, optional (default=None)
The number of features to consider when looking for the best split:
If int, then consider max_features features at each split.
If float, then max_features is a percentage and int(max_features * n_features) features are considered at each split.
If “auto”, then max_features=sqrt(n_features).
If “sqrt”, then max_features=sqrt(n_features).
If “log2”, then max_features=log2(n_features).
If None, then max_features=n_features.
Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.
random_state : int, RandomState instance or None, optional (default=None)
If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
max_leaf_nodes : int or None, optional (default=None)
Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
min_impurity_decrease : float, optional (default=0.)
A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
The weighted impurity decrease equation is the following:
N_t / N * (impurity - N_t_R / N_t * right_impurity
                    - N_t_L / N_t * left_impurity)
where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.
N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.
New in version 0.19.
min_impurity_split : float,
Threshold for early stopping in tree growth. A node will split if its impurity is above the threshold, otherwise it is a leaf.
Deprecated since version 0.19: min_impurity_split has been deprecated in favor of min_impurity_decrease in 0.19 and will be removed in 0.21. Use min_impurity_decrease instead. 
class_weight : dict, list of dicts, “balanced” or None, default=None
Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.
Note that for multioutput (including multilabel) weights should be defined for each class of every column in its own dict. For example, for four-class multilabel classification weights should be [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of [{1:1}, {2:5}, {3:1}, {4:1}].
The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
For multi-output, the weights of each column of y will be multiplied.
Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
'''
# sknn : scikit-learn nearest neighbour
sknn_params={}  # should not making this too complicated
sknn_params['n_neighbors'] = [1,5,10]
sknn_params['weights'] = ['uniform','distance']
sknn_params['algorithm'] = ['ball_tree','kd_tree']
sknn_params['max_features'] = ['auto']
sknn_params['p'] = [1,2]
sknn_params['metric'] = ['euclidean','chebyshev']
sknn_params['n_jobs'] = [5]
'''
n_neighbors : int, optional (default = 5)
Number of neighbors to use by default for kneighbors queries.
weights : str or callable, optional (default = ‘uniform’)
weight function used in prediction. Possible values:
‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.
‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
[callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.
algorithm : {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, optional
Algorithm used to compute the nearest neighbors:
‘ball_tree’ will use BallTree
‘kd_tree’ will use KDTree
‘brute’ will use a brute-force search.
‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.
Note: fitting on sparse input will override the setting of this parameter, using brute force.
leaf_size : int, optional (default = 30)
Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.
p : integer, optional (default = 2)
Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
metric : string or callable, default ‘minkowski’
the distance metric to use for the tree. The default metric is minkowski, and with p=2 is equivalent to the standard Euclidean metric. See the documentation of the DistanceMetric class for a list of available metrics.
metric_params : dict, optional (default = None)

http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.DistanceMetric.html#sklearn.neighbors.DistanceMetric

Additional keyword arguments for the metric function.
n_jobs : int, optional (default = 1)
The number of parallel jobs to run for neighbors search. If -1, then the number of jobs is set to the number of CPU cores. Doesn’t affect fit method.
'''
# sksvm : scikit-learn support vector machine  
sksvm_params={}  # linear/non-linear get about 10 of this
sksvm_params['C'] = [0.1,1,10]
sksvm_params['kernel'] = ['linear','poly', 'bf', 'sigmoid']
sksvm_params['degree'] = [3]
sksvm_params['gamma'] = ['auto']
sksvm_params['class_weight'] = ['balanced']
'''
C : float, optional (default=1.0)
Penalty parameter C of the error term.
kernel : string, optional (default=’rbf’)
Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples).
degree : int, optional (default=3)
Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.
gamma : float, optional (default=’auto’)
Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.
coef0 : float, optional (default=0.0)
Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’.
probability : boolean, optional (default=False)
Whether to enable probability estimates. This must be enabled prior to calling fit, and will slow down that method.
shrinking : boolean, optional (default=True)
Whether to use the shrinking heuristic.
tol : float, optional (default=1e-3)
Tolerance for stopping criterion.
cache_size : float, optional
Specify the size of the kernel cache (in MB).
class_weight : {dict, ‘balanced’}, optional
Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one. The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
verbose : bool, default: False
Enable verbose output. Note that this setting takes advantage of a per-process runtime setting in libsvm that, if enabled, may not work properly in a multithreaded context.
max_iter : int, optional (default=-1)
Hard limit on iterations within solver, or -1 for no limit.
decision_function_shape : ‘ovo’, ‘ovr’, default=’ovr’
Whether to return a one-vs-rest (‘ovr’) decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2).
Changed in version 0.19: decision_function_shape is ‘ovr’ by default.
New in version 0.17: decision_function_shape=’ovr’ is recommended.
Changed in version 0.17: Deprecated decision_function_shape=’ovo’ and None.
random_state : int, RandomState instance or None, optional (default=None)
The seed of the pseudo random number generator to use when shuffling the data. If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.
'''

# sknbb : scikit-learn naive bayes Bernoulli features (multivariate Bernoulli models)
sknbb_params={}
sknbb_params['alpha']=[0.5,1,2]
sknbb_params['fit_prior']=[True, False]
'''
alpha : float, optional (default=1.0)
Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
binarize : float or None, optional (default=0.0)
Threshold for binarizing (mapping to booleans) of sample features. If None, input is presumed to already consist of binary vectors.
fit_prior : boolean, optional (default=True)
Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
class_prior : array-like, size=[n_classes,], optional (default=None)
Prior probabilities of the classes. If specified the priors are not adjusted according to the data.
'''
# sknbg : scikit-learn naive bayes Gaussian features #
sknbg_params={}
'''
priors : array-like, shape (n_classes,)
Prior probabilities of the classes. If specified the priors are not adjusted according to the data.
'''
# sknbm : scikit-learn naive bayes Multinomial features #
sknbm_params={}
sknbm_params['alpha']=[0.5,1,2]
'''
	
alpha : float, optional (default=1.0)
Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
fit_prior : boolean, optional (default=True)
Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
class_prior : array-like, size (n_classes,), optional (default=None)
Prior probabilities of the classes. If specified the priors are not adjusted according to the data.
'''
# parameter collection
param_collection_sk_default=[skrf_params,sket_params,sknn_params,sksvm_params]
param_collection_names_sk_default=['skrf','sket','sknn','sksvm']



