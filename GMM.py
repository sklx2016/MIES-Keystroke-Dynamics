# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 22:35:41 2015

@author: vikiboy
"""
#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
from sklearn.externals import joblib

train=pd.read_csv('train.csv',header=0)

#Separate out X_train and y_train

X_train=train
y_train=train

X_train=X_train.drop([''],axis=1) #remove the output label column
y_train=y_train.drop([''],axis=1) #remove all the columns except the output label

X_train=X_train.values
y_train=y_train.values

#GMM 
n_classes=len(np.unique(y_train)) # should be 5

classifiers = dict((covar_type, GMM(n_components=n_classes,
                    covariance_type=covar_type, init_params='wc', n_iter=20))
                   for covar_type in ['spherical', 'diag', 'tied', 'full'])
                       
n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    #Initializing the Means of the GMM manually
    classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                                  for i in xrange(n_classes)])
    #Fitting the training data
    classifier.fit(X_train)
    #Predictions
    y_train_pred = classifier.predict(X_train)   
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    print "The accuracy in training is ",train_accuracy,"\n"    
    joblib.dump(classifier,'GMMclassifer.pkl')
    

    
                       
           
