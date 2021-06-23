#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

#%%
df = pd.read_csv("boston.csv")

#%%
df

#%%
sns.set(context="notebook")
cols = ["LSTAT", "RM", "MEDV"]
sns.pairplot(df[cols], height=2)
plt.show()

#%%
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(
    cm,
    cbar=True,
    annot=True,
    square=True,
    fmt=".2f",
    annot_kws={"size": 15},
    yticklabels=cols,
    xticklabels=cols,
)
plt.show()

#%%

from sklearn.linear_model import LinearRegression
sk_model = LinearRegression()
X = df[['RM']].values
y = df[['MEDV']].values

sk_model.fit(X,y)
#%%
sk_model.intercept_

#%%
def Regression_plot(X,y,model):
    plt.scatter(X,y,c='blue')
    plt.plot(X,model.predict(X),color='red')
    return None

Regression_plot(X,y,sk_model)
plt.xlabel('RM')
plt.ylabel('House Price')
plt.show()


