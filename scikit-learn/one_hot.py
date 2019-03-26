import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("./ml_book_code/data/adult.data", header=None, index_col=False, names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
'marital-status', 'occupation', 'relationship', 'race', 'gender',
'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
'income'])

# print(data.head())
# print(data.gender.value_counts())
""""
print("Original features:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
# print("Features after get_dummies:\n", list(data_dummies.columns))
# print(data_dummies.head())

features = data_dummies.ix[:,'age':'occupation_ Transport-moving']
X = features.values 
y = data_dummies['income_ >50K'].values 
# print("X.shape: {} y.shape: {}".format(X.shape, y.shape))

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print("Test score: {:.2f}".format(logreg.score(X_test, y_test)))
"""

print("Testing one_hot_encoder scikit-learn")
# create a DataFrame with an integer feature and a categorical string feature
demo_df = pd.DataFrame({'Integer Feature': [0, 1, 2, 1], 'Categorical Feature': ['socks', 'fox', 'socks', 'box']})
print(demo_df)
# print(pd.get_dummies(demo_df))
