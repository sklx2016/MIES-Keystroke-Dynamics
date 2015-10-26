# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 00:51:11 2015

@author: vikiboy
"""
#!/usr/bin/env python

import pandas as pd
from sklearn.externals import joblib

#Feature Extraction part
#Output - array of input (n,N_features)
X_test=pd.read_csv('test.csv',header=0)
X_test=X_test.values

GMM_classifier=joblib.load('GMMClassifier.pkl')
y_test=GMM_classifier.predict(X_test)