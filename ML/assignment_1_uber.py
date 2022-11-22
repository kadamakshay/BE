#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

import warnings 
warnings.filterwarnings("ignore")

#%%
df = pd.read_csv(r"C:\Users\Dell\OneDrive\Desktop\BT and ML\ML\uber.csv")
df.head(10)

#%%
#info of the data
df.info()

#%%
#statistical info of the data
df.describe()

#%%
#check for null values
df.isnull().sum()

#%%
#drop null values 
df = df.dropna()
df.isnull().sum()

#%%
#data types
df.dtypes

#%% columns 
df.columns

#%%
#cols_to_drop
df = df.drop(['Unnamed: 0',"key"],axis=1)
df.head()

#%%

plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),annot=True,yticklabels=True)
plt.show()

#%%

cols = df.columns.to_list()
cols.remove("pickup_datetime")
cols.remove("fare_amount")
cols

#%%
plt.figure(figsize=(14,7))
for i in range(len(cols)):
    plt.subplot(2,3,i+1)
    sns.boxplot(df[str(cols[i])])
plt.show()


#%%
#distribution
plt.figure(figsize=(14,7))
for i in range(len(cols)):
    plt.subplot(2,3,i+1)
    sns.distplot(df[str(cols[i])])
plt.show()
    

#%%

#zscore = x-mean/std

df['zscore_pickup_longitude'] = (df.pickup_longitude - df.pickup_longitude.mean()) / df.pickup_longitude.std()
df['zscore_pickup_latitude'] = (df.pickup_latitude - df.pickup_latitude.mean()) / df.pickup_latitude.std()
df['zscore_dropoff_longitude'] = (df.dropoff_longitude - df.dropoff_longitude.mean()) / df.dropoff_longitude.std()
df['zscore_dropoff_latitude'] = (df.dropoff_latitude - df.dropoff_latitude.mean()) / df.dropoff_latitude.std()
df['zscore_passenger_count'] = (df.passenger_count - df.passenger_count.mean()) / df.passenger_count.std()

df.head()

#%%
z_score_cols = ["zscore_pickup_longitude",'zscore_pickup_latitude','zscore_dropoff_longitude','zscore_dropoff_latitude','zscore_passenger_count']



#%%
#detect otliers
print("Outlier Data :\n ")
for i in range(0,len(z_score_cols)):
    outliers_data = df[(df[z_score_cols[i]] <-3) | (df[z_score_cols[i]]>3)]
    
outliers_data.head()


#%%

#remove outliers
for i in range(0,len(z_score_cols)):
    df1 = df[(df[z_score_cols[i]] >-3) & (df[z_score_cols[i]]<3)]
    
    
df = df1.copy()
df.head()


#%%
sc_x = StandardScaler()
df[cols] = sc_x.fit_transform(df[cols])
print(df.head())

#%%
#split the dataset into training and testring set
X = df[cols]
y = df['fare_amount']

#%%
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25, random_state=4200000)
print("length of X: ",len(X))
print("length of X_train: ",len(x_train))
print("length of X test: ",len(x_test))
print("length of Y: ",len(y))
print("length of y_train: ",len(y_train))
print("length of y_test: ",len(y_test))

#%%
lr = LinearRegression()
lr.fit(x_train,y_train)
pred = lr.predict(x_test)
mse = mean_squared_error(y_test,y_pred=pred)
rmse = np.sqrt(mse)
print("RMSE: ", rmse)

#%%
lr_r2_score_ = r2_score(y_test,pred)
print("r2 score: ", lr_r2_score_)


#%%

#RandomForest Regressor 

rf = RandomForestRegressor()
rf.fit(x_train,y_train)
pred = rf.predict(x_test)
rf_mse = mean_squared_error(y_test,y_pred=pred)

rf_rmse = np.sqrt(rf_mse)
print("RMSE: ", rf_rmse)


#%%
from sklearn.metrics import r2_score
rf_r2_score = r2_score(y_test,pred)
print("r2 score: ", rf_r2_score)


#%% Comparison

data={"RandomForest":[rf_rmse,rf_r2_score],"LinearRegression":[rmse,lr_r2_score_]}
comparison_df = pd.DataFrame(data=data, index=["RMSE","R2_Score"])

print(comparison_df)

