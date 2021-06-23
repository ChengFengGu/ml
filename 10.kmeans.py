#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.datasets import load_iris
import seaborn as sns


# In[2]:


iris = load_iris()
iris


# In[3]:


X = iris.data[:, 2:4]


# In[ ]:

sns.scatterplot(X[:, 0], X[:, 1], marker="o", label="see")
plt.show()

#%%
estimator = KMeans(n_clusters=3)
estimator.fit(X)
label_pred = estimator.labels_

#%%
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]

plt.scatter(x0[:, 0], x0[:, 1], c="red", marker="o", label="label1")
plt.scatter(x1[:, 0], x1[:, 1], c="blue", marker="*", label="label2")
plt.scatter(x2[:, 0], x2[:, 1], c="green", marker="+", label="label3")
plt.show()
