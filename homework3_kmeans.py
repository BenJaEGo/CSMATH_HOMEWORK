# author benjaego

import numpy as np
import matplotlib.pyplot as plt

_floatX = np.float32
_intX = np.int8

class kmeans(object):

    def __init__(self, k, m, n):
        self.cluster_number = k
        self.sample_number = m
        self.feature_number = n

        self.cluster_centers = np.zeros(shape=[self.cluster_number, self.feature_number], dtype=_floatX)
        self.cluster_label = np.zeros(shape=[self.sample_number, ], dtype=_floatX)

        self.data = np.zeros(shape=[self.sample_number, self.feature_number], dtype=_floatX)
        self.label = np.zeros(shape=[self.sample_number, ], dtype=_intX)

    def generate_data(self):
        x = np.random.randint(low=1, high=10, size=self.cluster_number)
        prob = x / x.sum()
        mean = np.random.randint(low=1, high=20, size=(self.cluster_number, self.feature_number))
        cov = np.eye(self.feature_number)

        for idx in range(self.sample_number):
            p = np.argmax(np.random.multinomial(n=1, pvals=prob, size=1))
            self.label[idx] = p
            self.data[idx, :] = np.random.multivariate_normal(mean[p, :], cov)

    def calc_dist_mat(self, x, y):
        xx = x * x
        yy = y * y
        xx_sum = np.sum(xx, axis=1)
        yy_sum = np.sum(yy, axis=1)
        xy = np.dot(x, y.T)
        return np.tile(xx_sum, [self.cluster_number, 1]).T + np.tile(yy_sum.T, [self.sample_number, 1]) - 2 * xy

    def calc_dist_loop(self, x, y):
        dist = np.zeros([self.sample_number, self.cluster_number])
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                dist[i, j] = np.linalg.norm(x[i, :] - y[j, :], ord=2) ** 2
        return dist

    def cluster(self, max_iter, eps):

        pre_loss = 0

        initial_centers = np.random.randint(self.sample_number, size=self.cluster_number)
        self.cluster_centers = self.data[initial_centers, :]

        for itr in range(max_iter):
            for cluster_idx in range(self.cluster_number):
                dist_mat = self.calc_dist_mat(self.data, self.cluster_centers)
                self.cluster_label = dist_mat.argmin(axis=1)
            for cluster_idx in range(self.cluster_number):
                data_slice = self.data[cluster_idx == self.cluster_label]
                self.cluster_centers[cluster_idx, :] = np.mean(data_slice, axis=0)

            loss = self.calc_loss()
            if np.abs(loss - pre_loss) < eps:
                break
            else:
                pre_loss = loss
                print('iteration %d, current cost is %f' % (itr, loss))

    def calc_loss(self):
        center_mat = self.cluster_centers[self.cluster_label, :]
        cost = (np.sum((self.data - center_mat) ** 2, axis=1) ** 1/2).sum()
        return cost

    def plot(self):
        assert(self.feature_number == 2)
        assert(self.cluster_number < 5)
        plt.figure(1)
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        for idx in range(self.sample_number):
            plt.sca(ax1)
            if 0 == self.cluster_label[idx]:
                plt.plot(self.data[idx, 0], self.data[idx, 1], 'yo')
            elif 1 == self.cluster_label[idx]:
                plt.plot(self.data[idx, 0], self.data[idx, 1], 'cx')
            elif 2 == self.cluster_label[idx]:
                plt.plot(self.data[idx, 0], self.data[idx, 1], 'g+')
            else:
                plt.plot(self.data[idx, 0], self.data[idx, 1], 'b*')
            plt.sca(ax2)
            if 0 == self.label[idx]:
                plt.plot(self.data[idx, 0], self.data[idx, 1], 'yo')
            elif 1 == self.label[idx]:
                plt.plot(self.data[idx, 0], self.data[idx, 1], 'cx')
            elif 2 == self.label[idx]:
                plt.plot(self.data[idx, 0], self.data[idx, 1], 'g+')
            else:
                plt.plot(self.data[idx, 0], self.data[idx, 1], 'b*')

        plt.sca(ax1)
        plt.title('kmeans results')
        plt.sca(ax2)
        plt.title('origin gaussian distribution')

        plt.show()

if __name__ == '__main__':

    k = 4
    m = 1000
    n = 2
    maxIter = 100
    epsilon = 1e-5

    k_obj = kmeans(k, m, n)
    k_obj.generate_data()

    # make sure calc_dist_mat is true
    dist1 = k_obj.calc_dist_loop(k_obj.data, k_obj.cluster_centers)
    dist2 = k_obj.calc_dist_mat(k_obj.data, k_obj.cluster_centers)
    print(np.allclose(dist1, dist2))

    k_obj.cluster(maxIter, epsilon)
    k_obj.plot()





