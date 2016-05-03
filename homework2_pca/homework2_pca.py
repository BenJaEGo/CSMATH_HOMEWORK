# author benjaego


import numpy as np
import matplotlib.pyplot as plt

_intX = np.int8

class PCA(object):

    def __init__(self, m, n):
        self.sample_number = m
        self.feature_number = n
        self.data = np.zeros(shape=[self.sample_number, self.feature_number + 1], dtype=_intX)

    def read_ophdds(self, file_dir):
        """
        Class:No of examples in training set
        0:  376
        1:  389
        2:  380
        3:  389
        4:  387
        5:  376
        6:  377
        7:  387
        8:  380
        9:  382
        """
        f = open(file_dir)
        data_list = list()
        for line in f.readlines():
            line_list = line.rstrip('\n').split(',')
            data_list.append(line_list)
        self.data = np.array(data_list, dtype=_intX)

    def sample(self, label):
        sample_data = self.data[self.data[:, -1] == label, :-1]
        return sample_data

    def substract_mean(self, mat):
        mat_mean = np.mean(mat, axis=0)
        mat_sub_mean = mat - mat_mean
        return mat_sub_mean

    def calc_cov_mat(self, mat):
        mat_cov = mat.T.dot(mat) / (mat.shape[0] - 1)
        return mat_cov

    def svd(self, mat):
        [u, s, v] = np.linalg.svd(mat)
        return [u, s, v]

    def generate_feature(self, mat, u, res=2):
        pca_feature = mat.dot(u[:, 0:res])
        return pca_feature

    def plot(self, feature, digit):
        plt.title('PCA result, digit %d' % digit)
        plt.plot(feature[:, 0], feature[:, 1], 'o')
        plt.xlabel('first principal component')
        plt.ylabel('second principal component')
        plt.show()


if __name__ == '__main__':

    m = 3823
    n = 64
    digit = 9
    pca_component = 2

    pca_obj = PCA(m, n)

    pca_obj.read_ophdds(r'optdigits.tra')
    mat = pca_obj.sample(digit)
    mat_sub = pca_obj.substract_mean(mat)
    mat_cov = pca_obj.calc_cov_mat(mat_sub)
    [u, s, v] = pca_obj.svd(mat_cov)

    feature = pca_obj.generate_feature(mat_sub, u, pca_component)

    # testify if the algorithm is correct by transform back
    whole_feature = pca_obj.generate_feature(mat_sub, u, res=mat.shape[1])
    raw_whole_feature = whole_feature.dot(u.T) + np.mean(mat, axis=0)
    print(np.allclose(raw_whole_feature, mat))

    pca_obj.plot(feature, digit)




