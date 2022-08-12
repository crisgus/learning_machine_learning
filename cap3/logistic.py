import numpy as np


class LogisticRegressionGD:
    """Perceptron clissifier.

    Parameters
    -----------
    - eta: float. Learning rate (between 0.00 and 1.00)
    - n_iter: int. Passes over the training data set.
    - random_state: int. Random number generator seed for random weight
        initialization.
    
    Attributes
    -----------
    - w_: 1d-array. Weights after fitting.
    - b_: scalar. Bias unit after fitting.

    - losses_: list. Mean squared error loss function values in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def activation(self, z):
        """Compute linear activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def net_input(self, X) -> float:
        """calculate the net input"""
        return np.dot(X, self.w_) + self.b_

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.5, 1, 0)

    def fit(self, X, y):
        """Fit training data
        
        Parameters
        ----------
        X: {array-like}, shape = [n_examples, n_features].
            Training vectors, where n_examples is the number of examples and
            n_feature is the number of feature.
        y: array-like, shape = [n-examples]

        Returns
        -------
        self: object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output))-(1-y).dot(np.log(1-output))) / X.shape[0]
            self.losses_.append(loss)
        return self
