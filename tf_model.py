import cv2
import numpy as np
import math
from sklearn.neural_network import MLPClassifier
from os.path import dirname, join, basename
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

character_scaler =  joblib.load("character_scaler.model")
character_dect =  joblib.load("character_dect.model")

img = cv2.imread(r'C:\Users\lx\Desktop\test_n.jpg')
res = cv2.resize(img, (16,36), interpolation=cv2.INTER_CUBIC)
res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
print( character_dect.predict(character_scaler.transform( [np.array( res ) .flatten() ] )) )


