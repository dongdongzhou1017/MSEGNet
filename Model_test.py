# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:06:35 2019

@author: Dongdong Zhou @JYU 
"""
from tensorflow.keras.models import load_model
import h5py
import pandas as pd 
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# The shape is (num_samples, sequence_length, num_channels,  load the test data and label 
x_test = h5py.File('.../test_data.h5', 'r')
x_test = x_test.get('X')
x_test = np.asarray(x_test, dtype=np.float16)
x_test = np.transpose(x_test, (0, 2, 1))


y_test = pd.read_csv('.../test_label.csv')

print('the shape of testing samples', x_test.shape)
print('the shape of testing label:', y_test.shape)

# load the saved model weight
model = load_model('.../model_name.h5')

# prediction 
pred = model.predict(x_test, batch_size=64, verbose=1)
predicted = np.argmax(pred, axis=1)
test_label_number = np.argmax(y_test, axis=1)
report = classification_report(y_test, predicted)
cm = confusion_matrix(y_test, predicted)
print(report)
print('Confusion matrix:',cm)

Acc = metrics.accuracy_score(y_test,predicted) 
print ('Test result:acc:%f'%(Acc))
#    



