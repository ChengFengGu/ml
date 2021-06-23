#%%
from numpy import *

m = 20

#%%

X0 = ones((m, 1))
X1 = arange(1, m + 1).reshape(m, 1)

#%%
X = hstack((X0, X1))
X

#%%
Y = array(
    [3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 12, 11, 13, 13, 16, 17, 18, 17, 19, 21]
).reshape(m, 1)
Y

#%%
alpha = 0.01

#%%
def cost_function(theta, X, Y):
    diff = dot(X, theta) - Y
    return (1 / 2 * m) * dot(diff.transpose(), diff)


#%%
def gradient_function(theta, X, Y):
    diff = dot(X, theta) - Y
    return (1 / m) * dot(X.transpose(), diff)


#%%
def gradient_descent(X, Y, alpha):
    theta = array([1, 1]).reshape(2, 1)
    gradient = gradient_function(theta, X, Y)
    while not all(abs(gradient) <= 1e-5):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, Y)
    return theta


#%%
optimal = gradient_descent(X, Y, alpha)
optimal 

#%%
print(f"cost function:{cost_function(optimal,X,Y)[0][0]}")

#%%
def plot(X,Y,theta):
    import matplotlib.pyplot as plt 
    ax = plt.subplot(111)
    ax.scatter(X,Y,s=30,c='blue')
    plt.xlabel('X')
    plt.ylabel('Y')

    x = arange(0,21,0.2)
    y = theta[0]  + theta[1]*x
    ax.plot(x,y)
    plt.show()
plot(X1,Y,optimal)

