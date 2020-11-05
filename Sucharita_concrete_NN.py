# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 14:19:11 2020

@author: SUCHARITA
"""
#Prepare a model for strength of concrete data using Neural Networks

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda

concrete =pd.read_csv("F:\\ExcelR\\Assignment\\Neural Network\\concrete.csv")
concrete.head(3)
concrete.columns
concrete.shape
concrete.describe() # varied range, standardization required
concrete.corr()
concrete.info() # float:8, int:1
concrete.isnull().sum()
concrete.boxplot(figsize=(30,10)) # there are outliers in slag, water,superplastic, finaagg, age,strength)
#handle outliers, replace by median, excluding strength from treatment
for col_name in concrete.columns[:-1]:
    q1 = concrete[col_name].quantile(0.25)
    q3 = concrete[col_name].quantile(0.75)
    iqr = q3 - q1
    
    low = q1-1.5*iqr
    high = q3+1.5*iqr
    concrete.loc[(concrete[col_name] < low) | (concrete[col_name] > high), col_name] = concrete[col_name].median() 

concrete.boxplot(figsize=(30,10)) # age and  fine aggregate have outliers
concrete["strength"].unique() # multiple values

# visualizations

sns.distplot(concrete['strength'])
sns.distplot(concrete['age'])

fig, ax2 = plt.subplots(3,3, figsize=(18,18))
sns.distplot(concrete['cement'],ax=ax2[0][0])
sns.distplot(concrete['slag'],ax=ax2[0][1])
sns.distplot(concrete['ash'],ax=ax2[0][2])
sns.distplot(concrete['water'],ax=ax2[1][0])
sns.distplot(concrete['superplastic'],ax=ax2[1][1])
sns.distplot(concrete['coarseagg'],ax=ax2[1][2])
sns.distplot(concrete['fineagg'],ax=ax2[2][0])
sns.distplot(concrete['age'],ax=ax2[2][1])
sns.distplot(concrete['strength'],ax=ax2[2][2])

# total dataset visualization in one plane

concrete.hist(figsize=(15,15))
sns.heatmap(concrete.corr(),annot= True)
sns.pairplot(concrete, diag_kind='kde')
plt.show()
print("Skewness = ",concrete['strength'].skew()) # 0.42
print("kurtosis = ",concrete['strength'].kurt()) # -0.31

# strength needs to be categorised

print((concrete['strength'].min())) # 2.33
print((concrete['strength'].max())) # 82.6
print((concrete['strength'].mean())) # 35.81
print((concrete['strength'].median())) # 34.35
print((concrete['strength'].std())) # 16.7
print(concrete.groupby('strength').groups) # multiple grps
print('Range of values: ', concrete['strength'].max()-concrete['strength'].min()) # 80.27
print('Range of values: ', concrete['cement'].max()-concrete['cement'].min()) # 438

# data scaled

min_d = concrete.min()
max_d = concrete.max()
normalized_df=(concrete - min_d)/(max_d - min_d)
normalized_df.head()
train,test = train_test_split(normalized_df, test_size = 0.20)
x_train = train.iloc[:,0:8]
y_train = train.iloc[:,8:9] 
x_test = test.iloc[:,0:8]
y_test = test.iloc[:,8:9]
y_train1= train["strength"]
len(x_train) # no of row:824
len(x_train.keys()) # no of columns: 8


# create keras model
model = Sequential()
model.add(Dense(5, input_dim=8, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))

# compile keras model, regression model
model.compile(loss='mean_squared_error',optimizer='rmsprop',metrics=['mean_squared_error'])

# fit keras model
model.fit(np.array(x_train),np.array(y_train), epochs=80, batch_size=10)
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

# evaluate keras mdoel accuracy
_,accuracy = model.evaluate(x_train, y_train)
print('Accuracy: %.2f' % (accuracy*100)) # 78%
pred_train = model.predict(np.array(x_train))
pred_train = pd.Series([i[0] for i in pred_train])
rmse_value = np.sqrt(np.mean((pred_train-y_train)**2))
plt.plot(pred_train,y_train,"bo")
np.corrcoef(pred_train,y_train1 )# we got high correlation 

 # evaluate on test data
_,accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100)) # 88%
pred_test = model.predict(np.array(x_test))
pred_test = pd.Series([i[0] for i in pred_test])
rmse_value_test = np.sqrt(np.mean((pred_test-y_test)**2))
plt.plot(pred_test,y_test,"rs")



###########################

from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=0)
x_resampled, y_resampled = smote_enn.fit_resample(x, y)
x_resampled.shape
y_resampled.shape
y_resampled["impact"].value_counts()
x_resampled["type_contact"].value_counts()
y_resampled["impact"].value_counts().plot(kind="pie") 
plt.show()


>>> from imblearn.combine import SMOTETomek
>>> smote_tomek = SMOTETomek(random_state=0)
>>> x1_resampled, y1_resampled = smote_tomek.fit_resample(x, y)
x1_resampled.shape
y1_resampled.shape
y1_resampled["impact"].value_counts()
x1_resampled["type_contact"].value_counts()


