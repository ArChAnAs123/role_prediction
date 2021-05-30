import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

data = pd.read_csv(r'C:\Users\Home\wbp\project\cdat.csv')
dt = data.iloc[:-1].values
label = data.iloc[:,-1] 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
df = data
label = df.iloc[:,-1]
original=label.unique() 
label=label.values
label2 = labelencoder.fit_transform(label)
y=pd.DataFrame(label2,columns=["ROLE"])
numeric=y["ROLE"].unique() 
y1 = pd.DataFrame({'ROLE':original, 'Associated Number':numeric})
X = data.iloc[:, :].values
X = pd.DataFrame(data,columns=['no_of_projects','coresub_skill','aptitude_skill','problemsolving_skill','programming_skill','abstractthink_skill',
                                'design_skill'])
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)#Decision tree
X_train2,X_test2,y_train2,y_test2=train_test_split(X,y,test_size=0.3,random_state=10)#XGBoost
X_train6,X_test6,y_train6,y_test6=train_test_split(X,y,test_size=0.2,random_state=15)#SVM


fs = SelectKBest(score_func=chi2, k='all')
fs.fit(X_train, y_train)
X_train_fs = fs.transform(X_train)
X_test_fs = fs.transform(X_test)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
def Dec_tree(X_train,y_train,X_test,y_test):
  from sklearn import tree
  clf = tree.DecisionTreeClassifier()
  clf = clf.fit(X_train, y_train)
  # Prediction
  y_pred = clf.predict(X_test)
  y_test_arr=y_test['role']
  from sklearn.metrics import confusion_matrix,accuracy_score
  accuracy = accuracy_score(y_test,y_pred)
from xgboost import XGBClassifier  
def xgboost(X_train,y_train,X_test,y_test,clf):
  
  shape = X_train.shape
  X_train=pd.to_numeric(X_train.values.flatten())
  X_train=X_train.reshape(shape)
 
  model = XGBClassifier()
  model.fit(X_train, y_train)
  xgb_y_pred  = clf.predict(X_test)
  from sklearn.metrics import confusion_matrix,accuracy_score
  y_pred = clf.predict(X_test)
  accuracy = accuracy_score(y_test,y_pred)
  xgb_accuracy = accuracy_score(y_test,xgb_y_pred)
  print("accuracy=",xgb_accuracy*100)
  return xgb_accuracy*100
a=['Business Analyst','Data Analyst','Software Developer','Software Tester','Technical Support','Technical Writer', 'UI/UX Designer', 'Web Developer', ]  
print("Boosting the Decision Tree  ")
acc = xgboost(X_train2,y_train2,X_test2,y_test2,clf)
x_new = [1, 1, 1, 2, 4, 2, 2]

new_pred  = clf.predict([x_new])
#print("Prediction : {}".format(y1[y1['Associated Number']==new_pred[0]]['ROLE']))
print("Prediction ",y1)
import pickle
with open('boost.pkl', 'wb') as file:
    pickle.dump(clf, file)

with open('boost.pkl', 'rb') as f:
        model = pickle.load(f)
       
print("pred",model.predict([np.array(x_new)]))

def prediction(x_new):
   y= model.predict(x_new)
   return a[y[0]]

