import pandas as pd
import numpy as np
import pickle


df=pd.read_csv(r'C:/Users/SAIDHANUSH/BankNote_Authentication.csv')

x=df.iloc[:,[0,1,2,3]]
y=df.iloc[:,[4]]

from sklearn.tree import DecisionTreeClassifier
tr=DecisionTreeClassifier()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
tr.fit(x_train,y_train)
pred=tr.predict(x_test)
from sklearn.metrics import accuracy_score
a=accuracy_score(pred,y_test)
pickle.dump(tr,open('note_model.pkl','wb'))