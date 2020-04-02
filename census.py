import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pyspark

df4= pyspark.csv.read('census.csv')

df= pd.read_csv('census.csv')
df= df.drop(['marital-status','relationship','race' ],
            axis=1)
df['education_level'].value_counts()
df.isnull().sum()

encoder= LabelEncoder()
df['income']= encoder.fit_transform(df['income'])
subset= df.select_dtypes(exclude=[np.number])
df= df.select_dtypes(np.number)
subset= pd.get_dummies(subset)

df= pd.concat([df, subset],axis=1, sort=False)


y= df['income']
df= df.drop('income', axis=1)
x= df.values
std= StandardScaler()
x= std.fit_transform(x)
x_train,x_test,y_train,y_test= train_test_split(x, y, test_size= .3)

#model= LogisticRegression()
#model = RandomForestClassifier(n_estimators=100)
model= xg_reg = xgb.XGBClassifier(subsample= 1.0,
                                 min_child_weight= 10,
                                 learning_rate= 0.1,
                                 gamma= 1.5,
                                 colsample_bytree= 1.0)
model.fit(x_train,y_train)
score= model.score(x_test, y_test)



import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,LeakyReLU
model= Sequential()
sgd = keras.optimizers.Adam(lr=0.001)
model.add(Dense(3,input_dim=103,kernel_initializer='he_normal', activation='relu'))
model.add(LeakyReLU(15))
model.add(LeakyReLU(30))
model.add(LeakyReLU(50))
model.add(Dense(1,activation='sigmoid' ))
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(x_train,y_train, epochs=100)
