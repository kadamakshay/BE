import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

#%%
data = pd.read_csv(r'C:\Users\Dell\OneDrive\Desktop\BT and ML\ML\diabetes.csv')

#%%
#statistical info 
data.describe()

#%%
#check imbalance dataset
sns.countplot(data.Outcome)
plt.show()

#%%
data.info()

#%%
data.isnull().sum()


#%%
data.dtypes

#%%
data.shape

#%%
#Split the data into training and testing set
x = data.drop(['Outcome'], 1)
y = data[['Outcome']]

#%%
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
print(data.head())

#%%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Training size of X train: {} ".format(len(x_train)))
print("Training size of X Test: {} ".format(len(x_test)))
print("Training size of Y Train: {} ".format(len(y_train)))
print("Training size of Y Test: {} ".format(len(y_test)))

#%%

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

#%%

y_pred = knn.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy is: ", acc)


#%%
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n", cm)

#%%
sns.heatmap(cm,annot=True)

#%%

#Error Rate : total number of two incorrect predictions (FN + FP) divided by the total number of a dataset (P + N).

err = (cm[0][1]+cm[1][0])/sum(sum(cm))
print("Error Rate: ", (cm[0][1]+cm[1][0])/sum(sum(cm)))


#%%
precision = cm[0][0]/(cm[0][0]+cm[0][1])
print("Precision: ", precision)

#%%
recall = cm[0][0]/(cm[0][0]+cm[1][0])
print("Recall: ", recall)
