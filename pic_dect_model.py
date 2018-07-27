from functools import reduce
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
# 用于判断框出来的图像是不是车牌
def flat(img):
    # print(np.array(img).flatten())
    return  np.array(img).flatten()

import math
# from handle import hog
from sklearn import cross_validation
import os
from sklearn.externals import joblib
# 读到plate里的图片 标记1   读到other图片 标记2

truedir = 'C:/Users/lx/Desktop/plate'
falsedir ='C:/Users/lx/Desktop/other'


train_key =list()
train_value =list()

for rt, dirs, files in os.walk(truedir):
    for i in files:
        filename = truedir+'/'+i
        img_gray = cv2.imread(filename )
        train_key.append(flat(img_gray))
        train_value.append(1)


for rt, dirs, files in os.walk(falsedir):
    for i in files:
        filename = falsedir+'/'+i
        img_gray = cv2.imread(filename)
        train_key.append(flat(img_gray))
        train_value.append(0)
scaler = StandardScaler()
scaler.fit(train_key)

mlp = MLPClassifier(solver='lbfgs', activation='relu', alpha=1e-5,hidden_layer_sizes=(60, 60), random_state=1)

svm = SVC(kernel='linear')

X_train, X_test, y_train, y_test = train_test_split( scaler.transform(train_key), train_value, test_size=0.3, random_state=33)
# svm.fit(X_train, y_train )
mlp.fit(X_train, y_train )
print(  len(y_test)  )
print( np.sum(y_test) )
print('score:',mlp.score(X_test,y_test))
# print(svm.predict(X_test))

# testdir = 'C:/Users/lx/Desktop/test1'
# for rt, dirs, files in os.walk(testdir):
#     for i in files:
#         filename = testdir+'/'+i
#         img_gray = cv2.imread(filename)
#         print(filename)
#         print(svm.predict([flat(img_gray)]) )
#
joblib.dump(scaler,'scaler.model')
joblib.dump(mlp, "dect.model")