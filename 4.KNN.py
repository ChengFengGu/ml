#%%
import pandas as pd
from scipy.sparse.construct import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#%%
# !wget https://garden-lu-oss.oss-cn-beijing.aliyuncs.com/data/fruit_data.csv


#%%
feat_cols = ["mass", "width", "height", "color_score"]
data = pd.read_csv("fruit_data.csv")

#%%
fruit2num = {"apple": 0, "mandarin": 1, "orange": 2, "lemon": 3}

#%%
data["label"] = data["fruit_name"].map(fruit2num)

#%%
data

#%%
X = data[feat_cols].values
y = data["label"].values

#%%
x_train_set, x_test_set, y_train_set, y_test_set = train_test_split(
    X, y, random_state=2021, test_size=0.2
)

#%%
X.shape[0], x_train_set.shape[0], x_test_set.shape[0], x_test_set.shape[0]

#%%

knn_model = KNeighborsClassifier(n_neighbors=5, weights="distance", p=2)
#%%

knn_model.fit(x_train_set, y_train_set)

#%%
accuracy = knn_model.score(x_test_set,y_test_set)
accuracy

#%%
num2fruit = dict(zip(fruit2num.values(),fruit2num.keys()))
num2fruit

#%%
