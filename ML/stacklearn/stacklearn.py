
''' handling model parameters '''

'''classification'''

if method == 'skrf':
    clf = sk.ensemble.RandomForestClassifier(**model_params[method])
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
    clf = xgb.train(model_params[method], xgb_data, num_boost_round=20)
    pred = clf.predict(xgb_data_test)











'''regression'''