# author benjaego

import numpy as np
import matplotlib.pyplot as plt

_floatX = np.float32

class CurveFitting(object):

    def __init__(self, n, m):
        self.n = n
        self.m = m + 1
        self.x = np.zeros((self.n, 1), dtype=_floatX)
        self.y = np.zeros((self.n, 1), dtype=_floatX)
        self.w = np.zeros((self.m, 1), dtype=_floatX)
        self.data = np.zeros((self.n, self.m), dtype=_floatX)

    def generate_data(self):
        interval = 1 / self.n
        self.x = np.arange(0, 1, interval).reshape(([self.n, 1]))
        self.y = np.sin(self.x * 2 * np.pi)
        for y in self.y:
            y += np.random.normal(0, 0.3)
        self.data = np.tile(self.x, reps=[1, self.m])
        for i in range(self.m):
            self.data[:, i] = self.data[:, i] ** i

    def regression_gradient_descent(self, max_iter, lr):

        for i in range(max_iter):
            h_theta_x = np.dot(self.data, self.w)
            loss = ((h_theta_x - self.y) ** 2).sum() / 2
            if i % 1000 == 0:
                print("current iter is: ...", i, "current loss is: ...", loss)
            self.w = self.w - lr * np.dot(self.data.T, (h_theta_x - self.y))

    def regression_normal_equation(self):
        self.w = np.linalg.inv(np.dot(self.data.T, self.data)).dot(self.data.T).dot(self.y)
        h_theta_x = np.dot(self.data, self.w)
        loss = ((h_theta_x - self.y) ** 2).sum() / 2
        print("using normal equation, loss is: ", loss)

    def plot(self):
        n_sample = 100
        interval = 1 / n_sample
        x_sample = np.arange(0, 1, interval).reshape(([n_sample, 1]))
        y_sample = np.sin(2 * x_sample * np.pi)

        data_sample = np.tile(x_sample, reps=[1, self.m])
        for i in range(self.m):
            data_sample[:, i] = data_sample[:, i] ** i
        h_sample = np.dot(data_sample, self.w)
        plt.plot(self.x, self.y, 'o', label='sample with gaussian noise')
        plt.plot(x_sample, y_sample, label='real sin(x)')
        plt.plot(x_sample, h_sample, label='fitted curve')
        plt.title('N=%s, M=%s' % (self.n, self.m-1))
        plt.legend()
        plt.show()

if __name__ == '__main__':
    max_iter = 100000
    lr = 0.1
    cf = CurveFitting(10, 9)
    cf.generate_data()
    # choice can be 'gd' or 'ne'
    choice = 'ne'
    if 'gd' == choice:
        cf.regression_gradient_descent(max_iter, lr)
    elif 'ne' == choice:
        cf.regression_normal_equation()
    else:
        print('you name it .')
    cf.plot()
