import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn import linear_model
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('Data.csv')

print(data.head())
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
print (data.describe())
print ("Skew is:", data.SalePrice.skew())
plt.hist(data.SalePrice, color='blue')
plt.show()
target = np.log(data.SalePrice)
print ("\n Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()
numeric_features = data.select_dtypes(include=[np.number])
corr = numeric_features.corr().round(2)
plt.rcParams['figure.figsize'] = (25, 20)
sns.heatmap(data=corr, annot=True, linewidths=.5)
correlated_features = set()
for i in range(len(corr.columns)):
for j in range(i):
if abs(corr.iloc[i, j]) > 0.8:
colname = corr.columns[i]
correlated_features.add(colname)

num_features=numeric_features
del num_features['Id']
del num_features['SalePrice']

fig, ax = plt.subplots(nrows=9, ncols=4)
i=0
for row in ax:
for col in row:
col.scatter(x=data[num_features.columns[i]], y=target)
i+=1
plt.show()
plt.rcParams['figure.figsize'] = (10, 6)

data.drop(labels=correlated_features, axis=1, inplace=True)

for i in data:
if is_numeric_dtype(data[i]):
if data[i].isnull().sum()>0:
data[i].fillna(data[i].median(),inplace=True)

le = LabelEncoder()
categorical_feature_mask = data.dtypes==object
categorical_cols = data.columns[categorical_feature_mask].tolist()

for i in categorical_cols:
try:
data[i]=le.fit_transform(data[i])
except TypeError:

del data[i]
continue

y = np.log(data.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
X, y, random_state=42, test_size=.33)
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
predictions = model.predict(X_test)

actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,color='b')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()
print("R^2 is:", r2_score(y_test,predictions))
print('RMSE is:', mean_squared_error(y_test, predictions))
print('RMSE is:', mean_absolute_error(y_test, predictions))