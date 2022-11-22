#%%

import pandas as pd 
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC, SVR

import warnings
warnings.filterwarnings("ignore")

#%%

df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\BT and ML\ML\spam.csv",encoding="ISO-8859-1")
df.head()

#%%

df = df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis="columns")
df.head()

#%%
df.isnull().sum()

#%%
df.columns = ['label','message']
df.head()

#%%

df.groupby('label').describe()

#%%

df.label = df.label.map({"ham":0,"spam":1})
df.head()


#%%
#split the data into training 

X = df[['message']]
y = df[['label']]

#%%

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.25, random_state=42)

print("Training size of X train: {} ".format(len(x_train)))
print("Training size of X Test: {} ".format(len(x_test)))
print("Training size of Y Train: {} ".format(len(y_train)))
print("Training size of Y Test: {} ".format(len(y_test)))


#%%
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.message)
x_train_count.toarray()[:2]


#%%
#convert test text data to vector
x_test_count = cv.transform(x_test.message)
x_test_count.toarray()[:2]

#%%
#convert test text data to vector
x_test_count = cv.transform(x_test.message)
x_test_count.toarray()[:2]

#%%

### Train KNN Model

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train_count,y_train)
knn_pred = knn.predict(x_test_count)


#%%

knn_cm = confusion_matrix(y_test,y_pred=knn_pred)
knn_cm

#%%

knn_acc = accuracy_score(y_test,y_pred=knn_pred)
knn_acc

#%%
 #### Train SVM Model
 
svm = SVC()
svm.fit(x_train_count,y_train)
svm_pred = svm.predict(x_test_count)

svm_cm = confusion_matrix(y_test,y_pred=svm_pred)

svm_acc = accuracy_score(y_test,y_pred=svm_pred)
print(svm_acc)

#%%
#Comparison

data = {"KNN Accuracy":knn_acc, "SVM Accuracy":svm_acc}
acc_ = pd.DataFrame(data=data,index=[1])
print(acc_)
