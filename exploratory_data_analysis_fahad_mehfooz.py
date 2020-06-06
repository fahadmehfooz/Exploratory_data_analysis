import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
%matplotlib inline 
sns.set(color_codes=True)

#reading dataset
data=pd.read_csv("AirQuality.csv")
data.head()
data.info

#removing ambiguity
data.replace({'state': {r'Uttaranchal': 'Uttarakhand', }}, regex = True, inplace = True)

#removing unnecessary columns
data.drop(['agency', 'location_monitoring_station','stn_code','sampling_date','type','location'], axis=1, inplace=True)

data.head(20)

#Removing duplicate rows
duplicate_rows = data[data.duplicated()]
duplicate_rows
data = data.drop_duplicates()
data.head(5)

data.isnull().sum()

#Replacing null values with mean
data['so2']=data['so2'].fillna(data['so2'].mean())
data['no2']=data['no2'].fillna(data['no2'].mean())
data['rspm']=data['rspm'].fillna(data['rspm'].mean())
data['spm']=data['spm'].fillna(data['spm'].mean())
data['pm2_5']=data['pm2_5'].fillna(data['pm2_5'].mean())

#Dropping date column
data.drop(['date'], axis=1, inplace=True)

data.shape

#visualizing for outliers
sns.boxplot(x=data['no2'])
sns.boxplot(x=data['rspm'])
sns.boxplot(x=data['spm'])

#remving outliers
q1=data.quantile(0.25)
q3=data.quantile(0.75)
IQR = q3-q1
data = data[~((data < (q1-1.5 * IQR)) |(data > (q3 + 1.5 * IQR))).any(axis=1)]

#staewise examples
data.state.value_counts().nlargest(40).plot(kind='bar', figsize=(10,5))

#finding corelation between gases
plt.figure(figsize=(20,10))
rel= data.corr()
sns.heatmap(rel,cmap='BrBG',annot=True)
rel

#so2 vs states
fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(data['so2'], data['state'])
ax.set_xlabel('so2')
ax.set_ylabel('state')
plt.show()

#no2 vs states
fig, ax = plt.subplots(figsize=(20,10))
ax.scatter(data['no2'], data['state'])
ax.set_xlabel('no2')
ax.set_ylabel('state')
plt.show()

#label encoding for states column
labelencoder = LabelEncoder()
data['state'] = labelencoder.fit_transform(data['state'])
data

#Defining x and y feature set
X = data[['so2', 'no2', 'rspm', 'spm', 'pm2_5']].values
X[0:5]
y = data['state'].values
y [0:5]

#normalizing data

X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

#Train, test split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

k = 20
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh

yhat = neigh.predict(X_test)
yhat[0:5]

print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))



