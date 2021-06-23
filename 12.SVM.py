#%%
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

#%%
boston = load_boston()

#%%
X, y = boston.data, boston.target
X, y

#%%
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#%%
from sklearn.svm import SVR

for kernel in ["linear", "rbf"]:
    svr = SVR(kernel=kernel)
    svr.fit(x_train, y_train)
    print(f"{kernel} || accuracy: {svr.score(x_test,y_test)}")


#%%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(x_train)

#%%
x_train_scalered = scaler.transform(x_train)
x_test_scalered = scaler.transform(x_test)

for kernel in ["linear", "rbf"]:
    svr = SVR(kernel=kernel, C=100, gamma=0.1)
    svr.fit(x_train_scalered, y_train)
    print(f"after preprocessed || {kernel} || {svr.score(x_test_scalered,y_test)}")
