import pandas as pd



'''example'''

train_df = pd.read_csv("train_2016.csv", parse_dates=["transactiondate"])      # parse_dates
'''
parse_dates : boolean or list of ints or names or list of lists or dict, default False
boolean. If True -> try parsing the index.
list of ints or names. e.g. If [1, 2, 3] -> try parsing columns 1, 2, 3 each as a separate date column.
list of lists. e.g. If [[1, 3]] -> combine columns 1 and 3 and parse as a single date column.
dict, e.g. {‘foo’ : [1, 3]} -> parse columns 1, 3 as date and call result ‘foo’
If a column or index contains an unparseable date, the entire column or index will be returned unaltered as an object data type. For non-standard datetime parsing, use pd.to_datetime after pd.read_csv
Note: A fast-path exists for iso8601-formatted dates.
'''
train_df.shape      # show data frame shapes
train_df.head()     # show top few data line

# plotting
plt.figure(figsize=(8,6))
plt.scatter(range(train_df.shape[0]), np.sort(train_df.logerror.values))    # scatter plot (x, y, ...)
plt.xlabel('index', fontsize=12)
plt.ylabel('logerror', fontsize=12)
plt.show()

# plotting by using percentile boundary of values
ulimit = np.percentile(train_df.logerror.values, 99)    # get percentile value
llimit = np.percentile(train_df.logerror.values, 1)     # get percentile value
train_df['logerror'].ix[train_df['logerror']>ulimit] = ulimit       # set dataframe.ix[index] = value
train_df['logerror'].ix[train_df['logerror']<llimit] = llimit       # set dataframe.ix[index] = value
plt.figure(figsize=(12,8))
sns.distplot(train_df.logerror.values, bins=50, kde=False)
plt.xlabel('logerror', fontsize=12)
plt.show()

# split all data by month
train_df['transaction_month'] = train_df['transactiondate'].dt.month        # datetime.dt.xyz  return xyz category (year/month/day/time/hour/minute/second/...) of datetime object
'''only usable when 'arse_dates' option is set when reading the corresponding column of the original data file'''
cnt_srs = train_df['transaction_month'].value_counts()      # count the number in each month
plt.figure(figsize=(12,6))
sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8, color=color[3])
plt.xticks(rotation='vertical')
plt.xlabel('Month of transaction', fontsize=12)
plt.ylabel('Number of Occurrences', fontsize=12)
plt.show()

# count the number of instance with resetting of index
(train_df['parcelid'].value_counts().reset_index())['parcelid'].value_counts()      # df.reset_index() can be seen as adding a new index to the df,

'''
note, most of the object return by panda is with df.index and df.values attributes
so, reset index is to set the index to the returning values (probably first column), and then put the data column label on the data

'''

'''example 2'''
prop_df = pd.read_csv("properties_2016.csv")
prop_df.shape


'''checking on missing values'''
missing_df = prop_df.isnull().sum(axis=0).reset_index()     # reset index
missing_df.columns = ['column_name', 'missing_count']       # add in column names
missing_df = missing_df.ix[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')

ind = np.arange(missing_df.shape[0])
width = 0.9
fig, ax = plt.subplots(figsize=(12,18))
rects = ax.barh(ind, missing_df.missing_count.values, color='blue')
ax.set_yticks(ind)
ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')
ax.set_xlabel("Count of missing values")
ax.set_title("Number of missing values in each column")
plt.show()


'''check out two dimensional distribution using sns.joinplot'''
plt.figure(figsize=(12,12))
sns.jointplot(x=prop_df.latitude.values, y=prop_df.longitude.values, size=10)
plt.ylabel('Longitude', fontsize=12)
plt.xlabel('Latitude', fontsize=12)
plt.show()


'''merge data, check column type, aggregate and count'''
train_df = pd.merge(train_df, prop_df, on='parcelid', how='left')
pd.options.display.max_rows = 65
dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df
dtype_df.groupby("Column Type").aggregate('count').reset_index()

'''some more basic usage'''
missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df['missing_ratio'] = missing_df['missing_count'] / train_df.shape[0]
missing_df.ix[missing_df['missing_ratio']>0.999]


'''univariate analysis, impute missing values with mean and calculate correlation with target variable'''
# Let us just impute the missing values with mean values to compute correlation coefficients #
mean_values = train_df.mean(axis=0)     # missing values in each category
train_df_new = train_df.fillna(mean_values, inplace=True)       # fillna

# Now let us look at the correlation coefficient of each of these variables #
x_cols = [col for col in train_df_new.columns if col not in ['logerror'] if train_df_new[col].dtype == 'float64']
'''get a lit of columns using this one line conditioning format'''

labels = []
values = []
for col in x_cols:
    labels.append(col)
    values.append(np.corrcoef(train_df_new[col].values, train_df_new.logerror.values)[0, 1])        # np.corrcoef(a,b)[0,1], the [0,1] part will get corr coefficient instead of the whole autocorrelation matrix
corr_df = pd.DataFrame({'col_labels': labels, 'corr_values': values})       # pd.DataFrame.from_dict(dict)  dict= {a: a_values, b: b_values}
corr_df = corr_df.sort_values(by='corr_values')

ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(12, 40))
rects = ax.barh(ind, np.array(corr_df.corr_values.values), color='y')
ax.set_yticks(ind)
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
# autolabel(rects)
plt.show()

'''check on whether a vairable is unique'''
corr_zero_cols = ['assessmentyear', 'storytypeid', 'pooltypeid2', 'pooltypeid7', 'pooltypeid10', 'poolcnt', 'decktypeid', 'buildingclasstypeid']
for col in corr_zero_cols:
    print(col, len(train_df_new[col].unique()))

'''correlation heatmap for selected variables'''
corr_df_sel = corr_df.ix[(corr_df['corr_values']>0.02) | (corr_df['corr_values'] < -0.01)]
corr_df_sel
cols_to_use = corr_df_sel.col_labels.tolist()
temp_df = train_df[cols_to_use]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(12, 12))
# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=1., square=True)
plt.title("Important variables correlation map", fontsize=15)
plt.yticks(rotation='horizontal')
plt.xticks(rotation='vertical')
plt.show()


'''count plot for certain column'''
plt.figure(figsize=(12,8))
sns.countplot(x="bathroomcnt", data=train_df)       # countplot | mean value will also be plotted after the fillna procedure
plt.ylabel('Count', fontsize=12)
plt.xlabel('Bathroom', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("Frequency of Bathroom count", fontsize=15)
plt.show()
'''distribution of the target variable w.r.t certain column'''
plt.figure(figsize=(12,8))
sns.boxplot(x="bathroomcnt", y="logerror", data=train_df)       # boxplot
plt.ylabel('Log error', fontsize=12)
plt.xlabel('Bathroom Count', fontsize=12)
plt.xticks(rotation='vertical')
plt.title("How log error changes with bathroom count?", fontsize=15)
plt.show()
'''
#violinplot
sns.violinplot(x='bedroomcnt', y='logerror', data=train_df)
'''


'''using ggplot for some visualizations'''
from ggplot import *
ggplot(aes(x='finishedsquarefeet12', y='taxamount', color='logerror'), data=train_df) + \
    geom_point(alpha=0.7) + \
    scale_color_gradient(low = 'pink', high = 'blue')


'''get feature importance using Extra tree regressor'''
train_y = train_df['logerror'].values
cat_cols = ["hashottuborspa", "propertycountylandusecode", "propertyzoningdesc", "fireplaceflag", "taxdelinquencyflag"]
train_df = train_df.drop(['parcelid', 'logerror', 'transactiondate', 'transaction_month']+cat_cols, axis=1)
feat_names = train_df.columns.values

from sklearn import ensemble
model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, max_features=0.3, n_jobs=-1, random_state=0)
model.fit(train_df, train_y)

## plot the importances ##
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()