# author: benjaego

import numpy as np
import matplotlib.pyplot as plt
_floatX = np.float32
_intX = np.int8

class SMO(object):

    def __init__(self, m, n, c, t, k_params):

        self.sample_number = m
        self.feature_number = n
        self.c = c
        self.t = t
        self.k_params = k_params

        self.data = np.zeros(shape=[self.sample_number, self.feature_number], dtype=_floatX)
        self.label = np.zeros(shape=[self.sample_number, 1], dtype=_intX)
        self.alpha = np.zeros(shape=[self.sample_number, 1], dtype=_floatX)
        self.b = 0
        self.k = np.zeros(shape=[self.sample_number, self.sample_number], dtype=_floatX)
        self.error_cache = np.zeros(shape=[self.sample_number, 2], dtype=_floatX)
        self.w = np.zeros(shape=[self.feature_number, 1], dtype=_floatX)

        self.p_label = np.zeros(shape=self.label.shape, dtype=_intX)

    def generate_data(self):
        mean_1 = [1] * self.feature_number
        mean_2 = [10] * self.feature_number
        cov = np.eye(self.feature_number)

        data = np.zeros(shape=[self.sample_number, self.feature_number], dtype=_floatX)
        label = np.zeros(shape=[self.sample_number, 1], dtype=_intX)
        for i in range(self.sample_number):

            if np.random.rand() > 0.5:
                data[i, :] = np.random.multivariate_normal(mean_1, cov)
                label[i, 0] = -1
            else:
                data[i, :] = np.random.multivariate_normal(mean_2, cov)
                label[i, 0] = 1
        self.data = data
        self.label = label
        self.k = self.calc_kernel_loop(self.data, self.data)

    # x and y are row vectors
    def lin_kernel(self, x, y):
        return np.dot(x, y.T)

    # x and y are row vectors
    def poly_kernel(self, x, y, c, d):
        return (np.dot(x, y.T) + c) ** d

    # x and y are row vectors
    def rbf_kernel(self, x, y, w):
        diff = x - y
        # L2 norm
        nu = np.linalg.norm(diff, ord=2) ** 2
        de = -2 * w ** 2
        return nu / de

    def calc_kernel_loop(self, x, y):
        mx, nx = x.shape
        my, ny = y.shape
        assert nx == ny
        k = np.zeros(shape=[mx, my])
        if 'lin' == self.k_params[0]:
            for x_idx in range(mx):
                for y_idx in range(my):
                    k[x_idx, y_idx] = self.lin_kernel(x[x_idx, :], y[y_idx, :])
            return k
        elif 'poly' == self.k_params[0]:
            for x_idx in range(mx):
                for y_idx in range(my):
                    k[x_idx, y_idx] = self.poly_kernel(x[x_idx, :], y[y_idx, :], self.k_params[1], self.k_params[2])
            return k
        elif 'rbf' == self.k_params[0]:
            for x_idx in range(mx):
                for y_idx in range(my):
                    k[x_idx, y_idx] = self.rbf_kernel(x[x_idx, :], y[y_idx, :], self.k_params[1])
            return k
        else:
            raise NotImplementedError('Kernel is not implement...')

    def calc_gx(self, idx):
        k_idx = self.k[:, idx].reshape(self.sample_number, 1)
        gx_idx = np.sum(self.alpha * self.label * k_idx) + self.b
        return gx_idx

    def calc_error(self, idx):
        error = self.calc_gx(idx) - self.label[idx]
        return error

    def calc_non_bound_alpha(self):
        lower_bound_idx = (self.alpha > 0).nonzero()
        upper_bound_idx = (self.alpha < self.c).nonzero()
        non_bound_idx = np.intersect1d(lower_bound_idx, upper_bound_idx)
        return non_bound_idx

    def take_step(self, idx1):
        error_idx1 = self.calc_error(idx1)

        if (error_idx1 * self.label[idx1] < -self.t and self.alpha[idx1] < self.c) \
           or (error_idx1 * self.label[idx1] > self.t and self.alpha[idx1] > 0):

            idx2, error_idx2 = self.select_second_alpha(idx1, error_idx1)
            alpha_idx1_old = self.alpha[idx1].copy()
            alpha_idx2_old = self.alpha[idx2].copy()

            if self.label[idx1] != self.label[idx2]:
                low = max(0, alpha_idx2_old - alpha_idx1_old)
                high = min(self.c, self.c + alpha_idx2_old - alpha_idx1_old)
            else:
                low = max(0, alpha_idx1_old + alpha_idx2_old - self.c)
                high = min(self.c, alpha_idx1_old + alpha_idx2_old)

            if low == high:
                print('low = high, nothing to do...')
                return 0

            eta = self.k[idx1, idx1] + self.k[idx2, idx2] - 2 * self.k[idx1, idx2]
            if eta < 0:
                print('woo, eta < 0 happens...')
                return 0
            self.alpha[idx2] = alpha_idx2_old + self.label[idx2] * (error_idx1 - error_idx2) / eta
            if self.alpha[idx2] > high:
                self.alpha[idx2] = high
            elif self.alpha[idx2] < low:
                self.alpha[idx2] = low

            self.error_cache[idx2] = [1, self.calc_error(idx2)]

            if np.abs(self.alpha[idx2] - alpha_idx2_old < 1e-5):
                print('idx2 not moving enough...')
                return 0

            self.alpha[idx1] = alpha_idx1_old + self.label[idx1] * self.label[idx2] * (alpha_idx2_old - self.alpha[idx2])
            self.error_cache[idx1] = [1, self.calc_error(idx1)]

            b_1 = -error_idx1 - self.label[idx1] * self.k[idx1, idx1] * (self.alpha[idx1] - alpha_idx1_old) - \
                self.label[idx2] * self.k[idx2, idx1] * (self.alpha[idx2] - alpha_idx2_old) + self.b
            b_2 = -error_idx2 - self.label[idx1] * self.k[idx1, idx2] * (self.alpha[idx1] - alpha_idx1_old) - \
                self.label[idx2] * self.k[idx2, idx2] * (self.alpha[idx2] - alpha_idx2_old) + self.b

            if 0 < self.alpha[idx1] < self.c:
                self.b = b_1
            elif 0 < self.alpha[idx2] < self.c:
                self.b = b_2
            else:
                self.b = (b_1 + b_2) / 2
            return 1
        else:
            return 0

    def select_second_alpha(self, idx1, error_idx1):

        self.error_cache[idx1] = [1, error_idx1]
        valid_error_cache = np.nonzero(self.error_cache[:, 0])[0]
        if len(valid_error_cache) > 1:
            idx2 = -1
            error_idx2 = 0
            max_error = 0
            for idx in valid_error_cache:
                if idx1 == idx:
                    continue
                error_idx = self.calc_error(idx)
                error = np.abs(error_idx1 - error_idx)
                if error > max_error:
                    idx2 = idx
                    error_idx2 = error_idx
                    max_error = error
            return idx2, error_idx2
        else:
            idx2 = idx1
            while idx2 == idx1:
                idx2 = int(np.random.uniform(0, self.sample_number))
            error_idx2 = self.calc_error(idx2)
            return idx2, error_idx2

    def start(self, max_iter):
        iter = 0
        entire_set = True
        alpha_pairs_changed = 0

        while iter < max_iter and (alpha_pairs_changed > 0 or entire_set):
            alpha_pairs_changed = 0
            if entire_set:
                for idx in range(self.sample_number):
                    alpha_pairs_changed += self.take_step(idx)
                    print("iterating entire training set, iter: %d, current index: %d, alpha pairs changed: %d"
                          % (iter, idx, alpha_pairs_changed))
                iter += 1
            else:
                non_bound_idx = self.calc_non_bound_alpha()
                for idx in non_bound_idx:
                    alpha_pairs_changed += self.take_step(idx)
                    print("iterating non-bound examples, iter: %d, current index: %d, alpha pairs changed: %d"
                          % (iter, idx, alpha_pairs_changed))
                iter += 1

            if entire_set:
                entire_set = False
            elif 0 == alpha_pairs_changed:
                entire_set = True

    def calc_weights(self):
        for idx in range(self.sample_number):
            self.w += (self.label[idx] * self.alpha[idx] * self.data[idx, :]).reshape(self.w.shape)

    def calc_train_error(self):
        for i in range(self.sample_number):
            self.p_label[i] = self.calc_gx(i)
        acc = np.sum(np.equal(self.p_label, self.label)) / self.sample_number
        print('final training accuracy is: %f' % acc)

    def plot(self):
        assert(self.feature_number == 2)
        plt.figure(1)
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)
        for idx in range(self.sample_number):
            plt.sca(ax1)
            if -1 == self.p_label[idx]:
                plt.plot(self.data[idx, 0], self.data[idx, 1], 'yo')
            elif 1 == self.p_label[idx]:
                plt.plot(self.data[idx, 0], self.data[idx, 1], 'cx')
            else:
                plt.plot(self.data[idx, 0], self.data[idx, 1], 'b*')
            plt.sca(ax2)
            if -1 == self.label[idx]:
                plt.plot(self.data[idx, 0], self.data[idx, 1], 'yo')
            elif 1 == self.label[idx]:
                plt.plot(self.data[idx, 0], self.data[idx, 1], 'cx')

        plt.sca(ax1)
        plt.title('svm results with parameters %s' % self.k_params)
        plt.sca(ax2)
        plt.title('origin gaussian distribution')

        plt.show()


if __name__ == '__main__':

    m = 2000
    n = 2
    c = 1
    t = 1e-3
	# choice can be 'lin', 'poly' or 'rbf'
    # choice = 'lin'
    # choice = 'poly'
    choice = 'rbf'
    k_params = []
    if 'lin' == choice:
        k_params = ['lin']
    elif 'poly' == choice:
        k_params = ['poly', 2, 3]
    elif 'rbf' == choice:
        k_params = ['rbf', 1.3]
    else:
        print('you name it ...')
    max_iter = 50

    smo_obj = SMO(m, n, c, t, k_params)
    smo_obj.generate_data()
    smo_obj.start(max_iter)
    smo_obj.calc_train_error()
    smo_obj.plot()






