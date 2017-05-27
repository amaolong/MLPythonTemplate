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


'''checking on and handling missing values'''
missing_df = prop_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
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








