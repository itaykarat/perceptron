from matplotlib import pyplot as plt


class plotter:
    def __init__(self, figsize=(10, 8), x_axis_title='feature 1', y_axis_title='feature 2',plot_title = 'Random Classification Data with 2 classes'):
        self.api = 'matplotlib'
        self.figsize = figsize
        self.x_axis_title = x_axis_title
        self.y_axis_title = y_axis_title
        self.plot_title = plot_title

    def plot_data(self,data):
        X, y = data
        # Plotting
        fig = plt.figure(figsize=self.figsize)
        plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'r^')
        plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
        plt.xlabel(self.x_axis_title)
        plt.ylabel(self.y_axis_title)
        plt.title(self.plot_title)
        plt.show()

    def plot_decision_boundary(self,data, theta):
        """
        :param data: tuple (X,y) where X is input and y is lable
        :param theta: parameters
        :return: None
        """

        X, y = data

        # The Line is y=ax+b
        # So, Equate ax+b = theta0.X0 + theta1.X1 + theta2.X2
        # Solving we find m and c
        x1 = [min(X[:, 0]), max(X[:, 0])]
        a = -theta[1] / theta[2]
        b = -theta[0] / theta[2]
        x2 = a * x1 + b

        # Plotting
        fig = plt.figure(figsize=self.figsize)
        plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], "r^")
        plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], "bs")
        plt.xlabel(self.x_axis_title)
        plt.ylabel(self.y_axis_title)
        plt.title(self.plot_title)
        plt.plot(x1, x2, 'y-')
        plt.show()
