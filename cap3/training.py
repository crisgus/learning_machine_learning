from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from cap2.region import plot_decision_regions
from logistic import LogisticRegressionGD


iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

# Let's separate now data into training set and test set
# We set test data as the 30% of total data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Let's standardize data
sc = StandardScaler()
# .fit find mu and sigma
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


class TrainingLogistic:

    def without_scikit(self):
        X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
        y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
        lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
        lrgd.fit(X_train_01_subset, y_train_01_subset)
        plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=lrgd)
        plt.xlabel('Petal lenght [standardized]')
        plt.ylabel('Petal width [standardized]')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
