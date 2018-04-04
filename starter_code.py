# -*- coding: utf-8 -*-
"""
Starter code for basic level
Colomb-ia

"""

#Importar librerías/Import libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from time import time
#features_train, features_test, labels_train, labels_test = preprocess()

#Leer datos/Read data
train = pd.read_csv("data/iris_train.csv")
test = pd.read_csv("data/iris_test.csv")

#Tratamiento de datos/Data processing
features_train = train[['Sepal length','Sepal width','Petal length','Petal width']]
labels_train =  train['Class']
features_test= test[['Sepal length','Sepal width','Petal length','Petal width']]
labels_test= test['Class']


#El función/The function
def randomForest(features_train,labels_train):
	clf=RandomForestClassifier(n_estimators=100,min_samples_leaf=20,max_features="auto")
	clf=clf.fit(features_train,labels_train)
	print "training time:", round(time()-t0, 3), "s"
	return clf


t0 = time()
clf=randomForest(features_train, labels_train)
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

accuracy = clf.score(features_test,labels_test)
print "Accuracy is", accuracy
