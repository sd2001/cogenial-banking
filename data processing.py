import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('E:\Gehra learning\ANN bank\Churn_Modelling.csv')
X=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values

from sklearn.preprocessing import LabelEncoder
labelencoder_X_1=LabelEncoder()
X[:,1]=labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2=LabelEncoder()
X[:,2]=labelencoder_X_1.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [1])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)

X = np.array(ct.fit_transform(X), dtype=np.float)
X=X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)


X_test=sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
classifier=Sequential()
classifier.add(Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X_train,y_train,batch_size=5,nb_epoch=100)

y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)



