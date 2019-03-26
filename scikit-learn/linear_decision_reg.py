from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import mglearn
import numpy as np

X,y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3,3,1000,endpoint=False).reshape(-1,1)

reg = DecisionTreeRegressor(min_samples_split=3).fit(X,y)
plt.plot(line, reg.predict(line), label="decision tree")

reg = LinearRegression().fit(X,y)
plt.plot(line, reg.predict(line), label="linear regression")

plt.plot(X[:,0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
# plt.show()
plt.close()

bins = np.linspace(-3,3,11)
which_bin = np.digitize(X, bins=bins)
# print("\nData points:\n", X[:5])
# print("\nBin membership for data points:\n", which_bin[:5])
encoder = OneHotEncoder(sparse=False)
encoder.fit(which_bin)
X_binned = encoder.transform(which_bin)
line_binned = encoder.transform(np.digitize(line, bins=bins))
# print(X_binned[:5])

X_combined = np.hstack([X,X_binned])
line_combined = np.hstack([line, line_binned])
print(X_combined)
reg = LinearRegression().fit(X_combined,y)
plt.plot(line, reg.predict(line_combined), label="Linear regression combined")
