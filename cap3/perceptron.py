from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
print(iris)
X = iris.data[:, [2,3]]
y = iris.target
