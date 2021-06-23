#%%
import numpy as np

data = np.array([[3, -1.7, 3.5, -6], [0, 4, -0.3, 2.5], [1, 3.5, -1.8, -4.5]])

#%%
print(data)

#%%
from sklearn.preprocessing import StandardScaler

data_standard_scaler = StandardScaler().fit_transform(data)

#%%
print(data_standard_scaler)

#%%
from sklearn.preprocessing import MinMaxScaler

data_minmax_scaler = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)

print(data_minmax_scaler)

#%%
from sklearn.preprocessing import Binarizer

data_binarizer = Binarizer().fit_transform(data)
print(data_binarizer)


