# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 23:31:10 2020

@author: SUCHARITA
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda

startup =pd.read_csv("F:\\ExcelR\\Assignment\\Neural Network\\50_Startups.csv")
startup.head(3)
startup.columns
startup.shape
startup.describe() # varied range, standardization required
startup.corr()
startup.info() # float:4, obj:1
startup.isnull().sum()
startup.boxplot(figsize=(30,10)) # profit has outlier and it is the target variables
#handle outliers, replace by median, excluding strength from treatment

startup["Profit"].unique() # multiple values

# visualizations

sns.distplot(startup["Profit"])

fig, ax2 = plt.subplots(2,2, figsize=(18,18))
sns.distplot(startup["Profit"],ax=ax2[0][0])
sns.distplot(startup["R&D Spend"],ax=ax2[0][1])
sns.distplot(startup["Administration"],ax=ax2[1][0])
sns.distplot(startup["Marketing Spend"],ax=ax2[1][1])

# total dataset visualization in one plane

startup.hist(figsize=(15,15))
sns.heatmap(startup.corr(),annot= True)
sns.pairplot(startup, diag_kind='kde')
plt.show()
print("Skewness = ",startup["Profit"].skew()) # 0.023
print("kurtosis = ",startup["Profit"].kurt()) # -0.06

# strength needs to be categorised

print((startup["Profit"].min())) # 14681.4
print((startup["Profit"].max())) # 192261.83
print((startup["Profit"].mean())) # 112012.63
print((startup["Profit"].median())) # 107978.19
print((startup["Profit"].std())) # 40306.18

# label encoding
string = ["State"]
from sklearn import preprocessing
n = preprocessing.LabelEncoder()
for i in string:
   startup[i] = n.fit_transform(startup[i])

# data scaled

min_d = startup.min()
max_d = startup.max()
normalized_df=(startup - min_d)/(max_d - min_d)
normalized_df.head()
train,test = train_test_split(normalized_df, test_size = 0.20)
x_train = train.iloc[:,0:4]
y_train = train.iloc[:,4:5] 
x_test = test.iloc[:,0:4]
y_test = test.iloc[:,4:5]
y_train1= train["Profit"] # series format
y_test1= test["Profit"] # series format
len(x_train) # no of row:40
len(x_train.keys()) # no of columns: 4


# create keras model
model = Sequential()
model.add(Dense(10, input_dim=4, activation='relu'))
model.add(Dense(5, activation='relu'))
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
plt.plot(pred_train,y_train,"bo") # positive corelation
np.corrcoef(pred_train,y_train1 )# we got high correlation 

 # evaluate on test data
_,accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100)) # 88%
pred_test = model.predict(np.array(x_test))
pred_test = pd.Series([i[0] for i in pred_test])
rmse_value_test = np.sqrt(np.mean((pred_test-y_test)**2))
plt.plot(pred_test,y_test,"rs")# positive corelation
np.corrcoef(pred_test,y_test1 )# we got high correlation 




