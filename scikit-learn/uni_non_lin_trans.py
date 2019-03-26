import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import Ridge 
from sklearn.model_selection import train_test_split

rnd = np.random.RandomState(0)
X_org = rnd.normal(size=(1000,3))
w = rnd.normal(size=3)

X = rnd.poisson(10 * np.exp(X_org))
y = np.dot(X_org,w)

# print("Number of feature appearances:\n{}".format(np.bincount(X[:, 0])))

bins = np.bincount(X[:,0])
plt.bar(range(len(bins)), bins, color='w')
plt.ylabel("Number of appearances")
plt.xlabel("Value")

# plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)
X_train = np.log(X_train + 1)
X_test = np.log(X_test + 1)

score = Ridge().fit(X_train, y_train).score(X_test, y_test)
print("Test Score {:.3f}".format(score))


