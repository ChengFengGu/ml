#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn


#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#%%

lensesLabels = ["age", "prescript", "astiqmatic", "tearRate", "class"]
lensesLabels

#%%
lenses = pd.read_table('lenses.txt',names=lensesLabels,sep='\t')
lenses

#%%
features = ["age", "prescript", "astiqmatic", "tearRate"]
X,y = lenses[features],lenses['class']
X,y

#%%
label_encoder = LabelEncoder()
for col in X.columns:
    X[col] = label_encoder.fit_transform(X[col])

#%%
X

#%%

y = label_encoder.fit_transform(y)

#%%
y

#%%
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.)

