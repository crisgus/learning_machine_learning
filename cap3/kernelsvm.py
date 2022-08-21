import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from region import plot_decision_regions


np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, 0)


class GenerateData:
    """Generates data we want to learn to"""

    def show(self):
        """Show data generated"""
        plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
                    c='royalblue', marker='s',
                    label='Class 1')
        plt.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1],
                    c='tomato', marker='o',
                    label='Class 0')
        plt.xlim([-3, 3])
        plt.ylim([-3, 3])
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()


class MySVM:
    """My class for SVM scikit learn SVM class"""

    def show(self):
        """Show data with boundary decision regions"""
        svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
        svm.fit(X_xor, y_xor)
        plot_decision_regions(X_xor, y_xor, classifier=svm)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    MySVM().show()
