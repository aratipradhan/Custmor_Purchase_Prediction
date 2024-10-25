# Importing the libraries
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

#import dataset
dataset = pd.read_csv(r"C:\Users\arati\Downloads\logit classification.csv")

X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#for this observation let me selcted as 100 observaion for test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm  = confusion_matrix(y_test, y_pred)
print("Confusion Matricx :",cm)

# This is to get the Models Accuracy 
from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print("Model Accuracy :",ac)

# This is to get the Classification Report
from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print("Classifiaction Table :",cr)

# This is to get the bias
bias = classifier.score(X_train, y_train)
print("Bias is :",bias)

# This is to get the Variance
variance = classifier.score(X_test, y_test)
print("Variance is :",variance)

#-----------------FUTURE PREDICTION 2------------

futuredataset = pd.read_csv(r"C:\Users\arati\DATAS SCIENCE NIT\OCTOBER\futuredata_2.csv")
future_data2 = futuredataset.copy()
#future_data2
futuredataset = futuredataset.iloc[:, 1:].values

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
M = sc.fit_transform(futuredataset)

y_pred2 = pd.DataFrame()

future_data2['y_pred2'] = classifier.predict(M)
future_data2.to_csv('pred_model2.csv')
# To get the path 
import os
os.getcwd()

# Save the trained model to disk

with open('model.pkl', 'wb') as model_file:
    pickle.dump(classifier, model_file)

# Save the scaler to disk
with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(sc, scaler_file) 


