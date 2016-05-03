# author benjaego

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

_floatX = np.float64

class GMM(object):

    def __init__(self, m, n, k):
        self.sample_number = m
        self.feature_number = n
        self.gaussian_number = k
        self.m_param = np.array(([1 / self.gaussian_number] * self.gaussian_number))
        self.g_param_mean = np.random.randint(low=1, high=5, size=(self.gaussian_number, self.feature_number))
        self.g_param_cov = np.zeros([self.gaussian_number, self.feature_number, self.feature_number], dtype=_floatX)
        for g_idx in range(self.gaussian_number):
            self.g_param_cov[g_idx] = np.eye(self.feature_number)

        self.w = np.zeros(shape=[self.sample_number, self.gaussian_number], dtype=_floatX)

        self.data = np.zeros(([self.sample_number, self.feature_number]), dtype=_floatX)
        self.label = np.zeros([self.sample_number, ], dtype=np.int8)

        self.p_label = np.zeros([self.sample_number, ], dtype=np.int8)

    def generate_data(self, m_param, g_param_mean, g_param_cov):

        for s_idx in range(self.sample_number):
            m_idx = np.argmax(np.random.multinomial(n=1, pvals=m_param, size=1))
            self.data[s_idx, :] = np.random.multivariate_normal(g_param_mean[m_idx, :], g_param_cov[m_idx, :, :])
            self.label[s_idx] = m_idx

    # todo()
    # need to be vectorized to speedup
    def calc_pxi_given_zi(self, xi, mean_zi, cov_zi):
        tmp1 = (2 * np.pi) ** (self.feature_number / 2)
        tmp2 = linalg.det(cov_zi)
        de = tmp1 * tmp2
        x_minus_mean = (xi - mean_zi).reshape(self.feature_number, 1)
        rhs = float(x_minus_mean.T.dot(np.linalg.inv(cov_zi)).dot(x_minus_mean))
        nu = np.exp(-1/2 * rhs)
        return nu / de

    def calc_pzi_given_xi(self, xi):
        px_given_z = np.zeros(shape=self.gaussian_number, dtype=_floatX)
        pxz = np.zeros(shape=self.gaussian_number, dtype=_floatX)
        for g_idx in range(self.gaussian_number):
            px_given_z[g_idx] = self.calc_pxi_given_zi(xi, self.g_param_mean[g_idx, :], self.g_param_cov[g_idx, :, :])
            pxz[g_idx] = px_given_z[g_idx] * self.m_param[g_idx]

        w = pxz / pxz.sum()

        return w

    def e_step(self):
        for s_idx in range(self.sample_number):
            self.w[s_idx, :] = self.calc_pzi_given_xi(self.data[s_idx, :])

    def m_step(self):
        self.m_param = np.mean(self.w, axis=0)
        for g_idx in range(self.gaussian_number):
            tmp1 = self.data - self.g_param_mean[g_idx, :]
            tmp2 = self.w[:, g_idx].reshape(self.sample_number, 1)
            tmp3 = tmp1 * tmp2
            nu = tmp3.T.dot(tmp1)
            de = np.sum(self.w[:, g_idx])
            self.g_param_cov[g_idx, :, :] = nu / de

        self.g_param_mean = self.w.T.dot(self.data) / np.sum(self.w, axis=0).reshape(self.gaussian_number, 1)

    def predict(self):
        for x_idx in range(self.sample_number):
            self.p_label[x_idx] = np.argmax(self.calc_pzi_given_xi(self.data[x_idx, :]))

    def calc_accuracy(self):
        self.predict()
        corrects = np.sum(np.equal(self.p_label, self.label))
        return corrects

    def plot(self):
        assert(self.feature_number == 2)
        assert(self.gaussian_number < 5)

        plt.figure(1)
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        for s_idx in range(self.sample_number):
            plt.sca(ax1)
            if 0 == self.p_label[s_idx]:
                plt.plot(self.data[s_idx, 0], self.data[s_idx, 1], 'yo')
            elif 1 == self.p_label[s_idx]:
                plt.plot(self.data[s_idx, 0], self.data[s_idx, 1], 'cx')
            elif 2 == self.p_label[s_idx]:
                plt.plot(self.data[s_idx, 0], self.data[s_idx, 1], 'g+')
            else:
                plt.plot(self.data[s_idx, 0], self.data[s_idx, 1], 'b*')
            plt.sca(ax2)
            if 0 == self.label[s_idx]:
                plt.plot(self.data[s_idx, 0], self.data[s_idx, 1], 'yo')
            elif 1 == self.label[s_idx]:
                plt.plot(self.data[s_idx, 0], self.data[s_idx, 1], 'cx')
            elif 2 == self.label[s_idx]:
                plt.plot(self.data[s_idx, 0], self.data[s_idx, 1], 'g+')
            else:
                plt.plot(self.data[s_idx, 0], self.data[s_idx, 1], 'b*')

        plt.sca(ax1)
        plt.title('gmm results')
        plt.sca(ax2)
        plt.title('origin gaussian distribution')

        plt.show()

if __name__ == '__main__':

    # k is gaussian number
    # m is sample number
    # n is feature number

    k = 4
    m = 3000
    n = 2

    prob = np.random.randint(low=1, high=10, size=k)
    m_param_true = prob / prob.sum()
    g_param_mean_true = np.random.randint(low=1, high=20, size=(k, n))
    g_param_cov_true = np.zeros(shape=(k, n, n), dtype=_floatX)
    for idx in range(k):
        cov = np.eye(n) * (idx + 2)
        g_param_cov_true[idx] = cov
    # print(m_param_true)
    # print(g_param_mean_true)
    # print(g_param_cov_true)

    gmm_object = GMM(m, n, k)
    gmm_object.generate_data(m_param_true, g_param_mean_true, g_param_cov_true)

    max_iter = 200
    for it in range(max_iter):
        gmm_object.e_step()
        gmm_object.m_step()

    # print(gmm_object.m_param)
    # print(gmm_object.g_param_mean)
    # print(gmm_object.g_param_cov)

    accuracy = gmm_object.calc_accuracy() / m
    print('accuracy is ...', accuracy)
    gmm_object.plot()
