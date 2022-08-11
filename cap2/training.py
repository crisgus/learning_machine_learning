import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cap2.perceptron import Perceptron
from cap2.adaline import AdalineGD, AdalineSGD


S = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
DF = pd.read_csv(S, header=None, encoding='utf-8')

y = DF.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

X = DF.iloc[0:100, [0, 2]].values
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

class PrinterData:

    def __init__(self):
        plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='Setosa')
        plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='s', label='Versicolor')
        plt.xlabel('Sepal lenght [cm]')
        plt.ylabel('Petal lenght [cm]')
        plt.legend(loc='upper left')
        plt.show()


class TrainingPerceptron:

    def __init__(self):
        ppn = Perceptron(eta=0.1, n_iter=10)
        ppn.fit(X, y)
        plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Number of updates')
        plt.show()


class TrainingAdaline:

    def raw(self):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
        ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
        ax1.plot(range(1, len(ada1.losses_)+1), np.log10(ada1.losses_), marker='o')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Mean squared error')
        ax1.set_title('Adaline - Learning rate 0.1')

        ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
        ax2.plot(range(1, len(ada2.losses_)+1), ada2.losses_, marker='o')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Mean squared error')
        ax2.set_title('Adaline - Learning rate 0.0001')

        plt.show()

    def standard(self):
        # Standardize data:
        # 
        # standard x_i = (x_i - mu_i) / sigma_i
        #
        ada_gd = AdalineGD(n_iter=20, eta=0.5)
        ada_gd.fit(X_std, y)
        plt.plot(range(1, len(ada_gd.losses_)+1), ada_gd.losses_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Mean squared error')
        plt.tight_layout()
        plt.show()
    
    def SGD(self):
        ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
        ada_sgd.fit(X_std, y)
        plt.plot(range(1, len(ada_sgd.losses_)+1), ada_sgd.losses_, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Average loss')
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    TrainingAdaline().SGD()
