#%%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.datasets import load_wine

#%%
wine = load_wine()


#%%
X = wine.data
y = wine.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021)

#%%
clf_tree = DecisionTreeClassifier(random_state=2021, criterion="entropy")
clf_forest = RandomForestClassifier(random_state=2021, criterion="entropy", n_estimators=25)

clf_tree = clf_tree.fit(x_train, y_train)
clf_forest = clf_forest.fit(x_train, y_train)

score_tree = clf_tree.score(x_test, y_test)
score_forest = clf_forest.score(x_test, y_test)

#%%
score_tree,score_forest


#%%
# tree_l = []
# forest_l = []
#
# for i in range(1,10):
#     rfc = RandomForestClassifier(n_estimators=25)
