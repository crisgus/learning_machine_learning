from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from cap2.region import plot_decision_regions


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

ppn = Perceptron(eta0=0.1, random_state=1)
# Let's our ppn learn
ppn.fit(X_train_std, y_train)
# Let's make our ppn predictions on our test
y_pred = ppn.predict(X_test_std)

# We introduce the metric for our leaning machine: accuracy.
# Accuracy is 1 - error
print('Misclassified examples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))

# Alternatively, score() combines predict() and accuracy_score()
print('Accuracy: %.3f' % ppn.score(X_test_std, y_test))

# Print regions
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier=ppn,
                      test_idx=range(105, 150))
plt.xlabel('Petal lenght [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
