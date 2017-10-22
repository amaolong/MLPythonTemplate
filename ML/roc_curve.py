

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