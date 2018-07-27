from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
import os
import cv2
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.externals import joblib
mlp = MLPClassifier(solver='lbfgs', activation='relu', alpha=1e-5,hidden_layer_sizes=(50, 50), random_state=1)
scaler = StandardScaler()
path ='E:/ALL_Char1004/ALL_Char'
train_key=[]
train_value=[]
for rt, dirs, files in os.walk(path):
    for i in files:
        try :
            pic_name = 'E:/ALL_Char1004/ALL_Char/'+i
            filetype= str(i[0]).upper()
            img = cv2.imread(pic_name)
            # print(pic_name)
            res = cv2.resize(img, (10, 8), interpolation=cv2.INTER_CUBIC)
            train_key.append(np.array(res).flatten())
            # print(filetype)
            train_value.append(filetype)
        except Exception:
            continue
scaler.fit(train_key)
X_train, X_test, y_train, y_test = train_test_split( scaler.transform(train_key), train_value, test_size=0.3, random_state=33)
mlp.fit(X_train, y_train )
print(  len(y_test)  )
print('score:',mlp.score(X_test,y_test))
joblib.dump(scaler,'character_scaler.model')
joblib.dump(mlp, "character_dect.model")
