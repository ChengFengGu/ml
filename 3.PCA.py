#%%
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine

#%%
wine = load_wine()
x = wine.data
y = wine.target

#%%
wine.feature_names


#%%
wine.target_names

#%%
import pandas as pd

sample = pd.concat([pd.DataFrame(x), pd.DataFrame(y)], axis=1)


#%%
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()

#%%
X_train_std = stdsc.fit_transform(x)
X_train_std

#%%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

#%%
reduced_x = pca.fit_transform(X_train_std)
reduced_x

#%%
red_x, red_y = [], []
blue_x, blue_y = [], []
green_x, green_y = [], []

for i in range(len(reduced_x)):
    if y[i] == 0:
        red_x.append(reduced_x[i][0])
        red_y.append(reduced_x[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_x[i][0])
        blue_y.append(reduced_x[i][1])
    else:
        green_x.append(reduced_x[i][0])
        green_y.append(reduced_x[i][1])

plt.scatter(red_x, red_y, c="r", marker="x")
plt.scatter(blue_x, blue_y, c="b", marker="D")
plt.scatter(green_x, green_y, c="g", marker=".")
plt.title("Wine Standard PCA")
plt.xlabel("Dimension1")
plt.ylabel("Dimension2")

plt.legend(wine.target_names, loc="best")
plt.show()

#%%
plt.matshow(pca.components_, cmap="plasma")
plt.yticks([0, 1], ["Dimension1", "Dimension2"])
plt.colorbar()

plt.xticks(range(len(wine.feature_names)), wine.feature_names, rotation=60, ha="left")
plt.show()

#%%


