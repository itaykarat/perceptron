import numpy as np
from Algo.activation_function import activation_function


class perceptron_algo:
    def __init__(self):
        self.activation_function = 'step function'


    def perceptron(self,data ,lr, epochs): # Perceptron as a Neural Net
        """
        :param data: tuple (X,y); X --> Inputs. ;  y --> labels/target.
        :param lr: learning rate.
        :param epochs: Number of iterations.
        :param m: number of training examples
        :param n: number of features
        :return:
        """
        X, y = data # unpack data tuple
        m, n = X.shape

        # Initializing parameters(theta) to zeros.
        # +1 in n+1 for the bias term.
        theta = np.zeros((n + 1, 1))

        # Empty list to store how many examples were
        # misclassified at every iteration.
        n_miss_list = []

        # Training.
        for epoch in range(epochs):

            # variable to store #misclassified.
            n_miss = 0

            # looping for every example.
            for idx, x_i in enumerate(X):

                # Insering 1 for bias, X0 = 1.
                x_i = np.insert(x_i, 0, 1).reshape(-1, 1)

                # Calculating prediction/hypothesis.
                y_hat = activation_function().step_func(np.dot(x_i.T, theta))

                # Updating if the example is misclassified.
                if (np.squeeze(y_hat) - y[idx]) != 0:
                    theta += lr * ((y[idx] - y_hat) * x_i)

                    # Incrementing by 1.
                    n_miss += 1

            # Appending number of misclassified examples
            # at every iteration.
            n_miss_list.append(n_miss)

        return theta, n_miss_list
