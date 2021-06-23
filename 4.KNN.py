#%%
import pandas as pd
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
data['label'] = data['fruit_name'].map(fruit2num)

#%%
data

#%%
X = data[feat_cols].values
y = data['label'].values

