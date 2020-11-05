# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:12:10 2020

@author: SUCHARITA
"""
# predict burnt area: size_category is thetarget variable, 0:small, 1:large
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix
# Importing necessary models for implementation of ANN
from keras.models import Sequential
from keras.layers import Dense #, Activation,Layer,Lambda
from sklearn.model_selection import train_test_split

forest = pd.read_csv("F://ExcelR//Assignment//Neural Network//forestfires.csv")
forest.head(3)
forest.columns
forest.shape
forest.describe()
forest.corr()


forest.info() # 20 int, 3 obj
forest["size_category"].unique() #['small', 'large']
forest['size_category'].value_counts() # 1:378 and 0 :139
forest['size_category'].value_counts().plot(kind="pie") # data imbalanced
forest['month'].value_counts().plot(kind="bar") # aug and sep have max entries
forest['day'].value_counts().plot(kind="bar") # almost equal mention
forest['wind'].value_counts().plot(kind="bar") # 2-4 range
forest['RH'].value_counts().plot(kind="pie")


plt.figure(figsize=(5,12))
sns.heatmap(forest.corr(),annot=True)
sns.pairplot(forest, diag_kind='kde')
plt.show()

forest.isnull().sum() # No missing values 
categorical= [var for var in  forest.columns if forest[var].dtype=='O']
print('There are {} categorical variables'.format(len(categorical)))
print('The categorical variables are:',categorical)
string_values=  ['month', 'day']
forest[categorical].isnull().sum()

from sklearn import preprocessing
n = preprocessing.LabelEncoder()
for i in string_values:
   forest[i] = n.fit_transform(forest[i])

forest['size_category'].replace({"small": 0,"large" : 1},inplace=True)
forest.info()

train,test = train_test_split(forest, test_size = 0.25)
x_train = train.iloc[:,0:30]
y_train = train.iloc[:,30:31] 
x_test = test.iloc[:,0:30]
y_test = test.iloc[:,30:31]

train.size_category.value_counts() 
test.size_category.value_counts() 

# handle imbalance data
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2) 
x_train_res, y_train_res = sm.fit_sample(x_train,y_train)
x_train_res.shape, y_train_res.shape 
x_train=x_train_res.copy()
y_train=y_train_res.copy()
y_train = pd.DataFrame(y_train)
#y_train.columns = y_train.columns.astype(str).str.replace("0", "size") # replace value of "0" as "size"
train1 = pd.concat((x_train,y_train),axis=1) # (574,31)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
MinMaxScaler(copy=True, feature_range=(0, 1))
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# create keras model
model = Sequential()
model.add(Dense(12, input_dim=30, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile keras model, classification model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit keras model
model.fit(np.array(x_train),np.array(y_train), epochs=150, batch_size=10)
model_loss = pd.DataFrame(model.history.history)
model_loss.plot()

# evaluate keras mdoel accuracy
_,accuracy = model.evaluate(x_train, y_train)
print('Accuracy: %.2f' % (accuracy*100)) # 97%
train_pred = model.predict_classes(x_train)
print(classification_report(y_train,train_pred)) # accuracy = 97%
print(confusion_matrix(y_train,train_pred))
#[[286   1]
# [ 16 271]]

# make predictions
test_pred = model.predict_classes(x_test)
print(classification_report(y_test,test_pred)) # accuracy = 88%
print(confusion_matrix(y_test,test_pred))
#[[88  3]
# [13 26]]


# checking again for model accuracy using other metrics
# pd.Series - > convert list format Pandas Series data structure
train_pred = pd.Series([i[0] for i in train_pred])

burnt_class = ["S","L"]
# converting series because add them as columns into data frame
pred_train_class = pd.Series(["S"]*574)
pred_train_class[[i>0.5 for i in train_pred]] = "L"

train1["original_class"] = "S"
train1.loc[train1.size_category==1,"original_class"] = "L"
train1.original_class.value_counts()

# Two way table format 
print(classification_report(pred_train_class,train1.original_class)) # 97%
print(confusion_matrix(pred_train_class,train1.original_class))
#[[271   1]
# [ 16 286]]

# we need to reset the index values of train data as the index values are random numbers
np.mean(pred_train_class==pd.Series(train1.original_class).reset_index(drop=True)) # 97%
pd.crosstab(pred_train_class,pd.Series(train1.original_class).reset_index(drop=True))
#original_class    L    S
#row_0                   
#L               271    1
#S                16  286



# Predicting for test data 
test_pred = model.predict_classes(x_test)
test_pred = pd.Series([i[0] for i in test_pred])
pred_test_class = pd.Series(["S"]*130)
pred_test_class[[i>0.5 for i in test_pred]] = "L"
test["original_class"] = "S"
test.loc[test.size_category==1,"original_class"] = "L"
test.original_class.value_counts() # S:91, L:39
print(classification_report(pred_test_class,test.original_class)) # 88%
print(confusion_matrix(pred_test_class,test.original_class))
#[[26  3]
# [13 88]]



temp = pd.Series(test.original_class).reset_index(drop=True)
np.mean(pred_test_class==pd.Series(test.original_class).reset_index(drop=True)) # 87.69%
len(pred_test_class==pd.Series(test.original_class).reset_index(drop=True))
confusion_matrix(pred_test_class,temp)
pd.crosstab(pred_test_class,pd.Series(test.original_class).reset_index(drop=True)).plot(kind ="bar")

# small picture -  ANN network and its layers 
from keras.utils import plot_model
plot_model(model,to_file="first_model.png")
