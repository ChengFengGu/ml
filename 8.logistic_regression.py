#%%

xdata = [[6000, 58], [9000, 77], [11000, 89], [15000, 54]]
ydata = [0,0,1,1]

#%%
from sklearn.linear_model import LogisticRegression
import numpy as np

#%%

X = np.array(xdata)
y = np.array(ydata)

#%%
log_r = LogisticRegression()
log_r.fit(X, y)


#%%
log_r.predict(np.array([[12000,60]]))[0]

