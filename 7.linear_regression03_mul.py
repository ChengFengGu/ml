#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, r2_score


#%%

data = np.loadtxt("ex1data2.txt", encoding="gbk", delimiter=",")
data

#%%
X = data[:, :2]
y = data[:, 2]
X, y

#%%
lr = LinearRegression().fit(X, y)


#%%
predict = lr.predict(X)
predict

#%%
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(X[:,0],X[:,1],predict,color='red')
ax.plot(X[:,0],X[:,1],predict,color='blue')

plt.show()
