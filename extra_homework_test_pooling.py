
import numpy as np
from scipy import signal
import scipy as sp
import time
from extra_homework_utils import *
from extra_homework_test_convolution import *

class PoolingLayer(object):

    # input_feature_maps [mini_batch_size, featureMapsNumbers, featureHeight, featureWidth]
    # pool_shape [poolingHeight, poolingWidth]
    # input_feature_maps_shape [featureMapsNumbers, featureHeight, featureWidth]
    def __init__(self, pool_shape, input_feature_maps_shape):
        self.pool_shape = pool_shape
        self.pool_shape_h = pool_shape[0]
        self.pool_shape_w = pool_shape[1]

        # print(input_feature_maps_shape)
        self.n_feature_maps = input_feature_maps_shape[0]
        self.delta_size_h = input_feature_maps_shape[1]
        self.delta_size_w = input_feature_maps_shape[2]
        self.pooled_h = input_feature_maps_shape[1] // self.pool_shape_h
        self.pooled_w = input_feature_maps_shape[2] // self.pool_shape_w
        # self.out_fea_shape = [self.n_feature_maps, self.pooled_h, self.pooled_w]

        # using for back propagation from softmax layer
        self.n_out = self.n_feature_maps * self.pooled_h * self.pooled_w

        # using for back propagation from convolve layer

    def calc_max_pooling_split(self, input_feature_maps):

        n_batch_size = input_feature_maps.shape[0]
        self.out_fea_maps = np.zeros([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w])
        self.max_pooling_index = np.zeros([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w])

        # split in width direction first
        temp = np.asarray(np.split(input_feature_maps, self.pooled_w, axis=3))
        # use transpose and reshape to retain the correct dimension sequence
        # pooled_w and pooled_shape_w are last two dimensions
        temp = temp.transpose(1, 2, 3, 0, 4).reshape([n_batch_size, self.n_feature_maps, input_feature_maps.shape[2], self.pooled_w, self.pool_shape_w])
        # split in height direction which is the 2nd dimension
        temp = np.asarray(np.split(temp, self.pooled_h, axis=2))
        # use tranpose and reshape to retain the correct dimension sequence again
        temp = temp.transpose(1, 2, 0, 4, 3, 5).reshape([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w, self.pool_shape_h, self.pool_shape_w])
        # compress the last two demension into one demension
        temp = temp.reshape([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w, np.prod(self.pool_shape[:])])

        # print(temp.shape)
        self.max_pooling_index = np.argmax(temp, axis=4)
        self.out_fea_maps = np.max(temp, axis=4)
        self.out_fea_maps_v = self.out_fea_maps.reshape(n_batch_size, self.n_out)

    def calc_max_pooling_loop(self, input_feature_maps):

        n_batch_size = input_feature_maps.shape[0]
        self.out_fea_maps = np.zeros([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w])
        self.max_pooling_index = np.zeros([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w])
        for imgInd in range(n_batch_size):
            for filtInd in range(self.n_feature_maps):
                for rInd in range(self.pooled_h):
                    for cInd in range(self.pooled_w):
                        rStart = self.pool_shape_h * rInd
                        rEnd = self.pool_shape_h * (rInd + 1)
                        cStart = self.pool_shape_w * cInd
                        cEnd = self.pool_shape_w * (cInd + 1)
                        pooled_patch = input_feature_maps[imgInd, filtInd, rStart:rEnd, cStart:cEnd]
                        self.out_fea_maps[imgInd, filtInd, rInd, cInd] = pooled_patch.max()
                        max_index = pooled_patch.argmax()
                        # max_index_row = max_index // pooled_patch.shape[1]
                        # max_index_col = max_index % pooled_patch.shap[1]
                        self.max_pooling_index[imgInd, filtInd, rInd, cInd] = max_index
        self.out_fea_maps_v = self.out_fea_maps.reshape(n_batch_size, self.n_out)

    def calc_mean_pooling_loop(self, input_feature_maps):

        n_batch_size = input_feature_maps.shape[0]
        self.out_fea_maps = np.zeros([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w])
        for imgInd in range(n_batch_size):
            for filtInd in range(self.n_feature_maps):
                for rInd in range(self.pooled_h):
                    for cInd in range(self.pooled_w):
                        rStart = self.pool_shape_h * rInd
                        rEnd = self.pool_shape_h * (rInd + 1)
                        cStart = self.pool_shape_w * cInd
                        cEnd = self.pool_shape_w * (cInd + 1)
                        pooled_patch = input_feature_maps[imgInd, filtInd, rStart:rEnd, cStart:cEnd]
                        # print(pooled_patch.shape)
                        self.out_fea_maps[imgInd, filtInd, rInd, cInd] = pooled_patch.mean()
        self.out_fea_maps_v = self.out_fea_maps.reshape(n_batch_size, self.n_out)

    def calc_mean_pooling_convolve(self, input_feature_maps):

        n_batch_size = input_feature_maps.shape[0]
        self.out_fea_maps = np.zeros([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w])
        # print(self.out_fea_maps.shape)
        for imgInd in range(n_batch_size):
            for filtInd in range(self.n_feature_maps):
                im = input_feature_maps[imgInd, filtInd, :, :]
                p_feat_conv = signal.correlate2d(im, np.ones(self.pool_shape), 'valid')
                # print(p_feat_conv.shape)
                factor = 1 / (self.pool_shape[0] * self.pool_shape[1])
                p_feat_conv = p_feat_conv[0::self.pool_shape[0], 0::self.pool_shape[1]]
                # print(p_feat_conv.shape)
                self.out_fea_maps[imgInd, filtInd, :, :] = factor * p_feat_conv
        self.out_fea_maps_v = self.out_fea_maps.reshape(n_batch_size, self.n_out)

    # pooling layer is followed by a hidden layer or a softmax layer
    # delta is the error term corresponding to downsample errors
    # delta_c is the error term corresponding to upsample errors backpropagate to convolve layer
    # delta_v is the vectorized error term in case next layer is a hidden or softmax layer
    def calc_delta_mean_v(self, next_layer_W, next_layer_delta):

        n_batch_size = next_layer_delta.shape[0]
        self.delta = np.zeros([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w])
        self.delta_c = np.zeros([n_batch_size, self.n_feature_maps, self.delta_size_h, self.delta_size_w])
        delta_v = np.dot(next_layer_delta, next_layer_W.transpose())
        self.delta = delta_v.reshape([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w])
        for imgInd in range(n_batch_size):
            for filtInd in range(self.n_feature_maps):
                self.delta_c[imgInd, filtInd, :, :] = (1 / np.prod(self.pool_shape[:])) * sp.linalg.kron(np.ones(self.pool_shape), self.delta[imgInd, filtInd, :, :])
        self.delta_v = self.delta.reshape([n_batch_size, self.n_out])

    # pooling layer is followed by a convolutional layer
    # next_layer_W [featureMapsNumber, filtersNumber, filterHeight, filterWidth]
    # next_layer_delta [n_batch_size, filtersNumber, featureHeight, featureWidth]
    # delta is the error term corresponding to downsample errors
    # delta_c is the error term corresponding to upsample errors backpropagate to convolve layer
    # delta_v is the vectorized error term in case next layer is a hidden or softmax layer
    def calc_delta_mean_c(self, next_layer_W, next_layer_delta):

        n_batch_size = next_layer_delta.shape[0]
        self.delta = np.zeros([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w])
        n_filters = next_layer_W.shape[1]

        # start_time = time.clock()
        for imgIdn in range(n_batch_size):
            for feaIdn in range(self.n_feature_maps):
                temp_delta = np.zeros([self.pooled_h, self.pooled_w])
                for filtIdn in range(n_filters):
                    temp_delta += signal.convolve2d(next_layer_delta[imgIdn, filtIdn, :, :], next_layer_W[feaIdn, filtIdn, :, :], 'full')
                self.delta[imgIdn, feaIdn, :, :] = temp_delta
        # self.delta_v = self.delta.reshape([n_batch_size, self.n_out])
        # end_time = time.clock()
        # print('compute delta in pooling layer from convolve layer takes:...%f' % (end_time-start_time))

        # start_time = time.clock()
        self.delta_c = np.zeros([n_batch_size, self.n_feature_maps, self.delta_size_h, self.delta_size_w])
        for imgInd in range(n_batch_size):
            for filtInd in range(self.n_feature_maps):
                self.delta_c[imgInd, filtInd, :, :] = (1 / np.prod(self.pool_shape[:])) * sp.linalg.kron(np.ones(self.pool_shape), self.delta[imgInd, filtInd, :, :])
        # end_time = time.clock()
        # print('upsample delta in pooling layer takes:...%f' % (end_time-start_time))

    def calc_delta_max_v(self, next_layer_W, next_layer_delta):

        n_batch_size = next_layer_delta.shape[0]
        self.delta = np.zeros([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w])
        self.delta_c = np.zeros([n_batch_size, self.n_feature_maps, self.delta_size_h, self.delta_size_w])
        delta_v = np.dot(next_layer_delta, next_layer_W.transpose())
        self.delta = delta_v.reshape([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w])
        self.delta_v = self.delta.reshape([n_batch_size, self.n_out])

        tmp_delta_c = self.delta_c.reshape(np.prod(self.delta.shape[:]), np.prod(self.pool_shape[:]))
        tmp_delta = self.delta.reshape(np.prod(self.delta.shape[:]), )
        tmp_index = self.max_pooling_index.reshape(np.prod(self.max_pooling_index.shape[:]), )
        tmp_delta_c[np.arange(tmp_delta_c.shape[0]), tmp_index] = tmp_delta
        self.delta_c = tmp_delta_c.reshape(self.delta_c.shape)


    def calc_delta_max_c(self, next_layer_W, next_layer_delta):

        n_batch_size = next_layer_delta.shape[0]
        self.delta = np.zeros([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w])
        n_filters = next_layer_W.shape[1]
        # start_time = time.clock()
        for imgIdn in range(n_batch_size):
            for feaIdn in range(self.n_feature_maps):
                temp_delta = np.zeros([self.pooled_h, self.pooled_w])
                for filtIdn in range(n_filters):
                    temp_delta += signal.convolve2d(next_layer_delta[imgIdn, filtIdn, :, :], next_layer_W[feaIdn, filtIdn, :, :], 'full')
                self.delta[imgIdn, feaIdn, :, :] = temp_delta
        self.delta_v = self.delta.reshape([n_batch_size, self.n_out])
        # end_time = time.clock()
        # print('calc_delta_max_c for convolve2d takes:...%f' % (end_time-start_time))

        # start_time = time.clock()
        self.delta_c = np.zeros([n_batch_size, self.n_feature_maps, self.delta_size_h, self.delta_size_w])
        tmp_delta_c = self.delta_c.reshape(np.prod(self.delta.shape[:]), np.prod(self.pool_shape[:]))
        tmp_delta = self.delta.reshape(np.prod(self.delta.shape[:]), )
        tmp_index = self.max_pooling_index.reshape(np.prod(self.max_pooling_index.shape[:]), )
        tmp_delta_c[np.arange(tmp_delta_c.shape[0]), tmp_index] = tmp_delta
        self.delta_c = tmp_delta_c.reshape(self.delta_c.shape)
        # end_time = time.clock()
        # print('part 1 takes:...%f' % (end_time-start_time))

def test_pooling():
    dataset = 'MNIST_DATASET.pkl.gz'
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    train_data = train_set_x[0:1000]
    test_data = train_set_y[0:1000]

    in_feature_shape = (1, 28, 28)
    filter_shape = (1, 3,  5, 5)

    n_batch_size = train_data.shape[0]
    height = 28
    width = 28

    in_feature_maps = train_data.reshape(n_batch_size, 1, height, width)

    convolve_layer = ConvolveLayer(filter_shape, in_feature_shape, 'rectified')
    convolve_layer.calc_convolve_correlate(in_feature_maps)
    output_feature_maps_correlate = convolve_layer.out_fea_maps

    pool_shape = [2, 2]
    in_feature_shape_pool = convolve_layer.out_fea_shape
    pooling_layer = PoolingLayer(pool_shape, in_feature_shape_pool)

    start_time = time.clock()
    pooling_layer.calc_mean_pooling_loop(output_feature_maps_correlate)
    output_feature_maps_mean_loop = pooling_layer.out_fea_maps
    end_time = time.clock()
    print('elapsed time for mean pooling loop method:... %f' % (end_time - start_time))

    start_time = time.clock()
    pooling_layer.calc_mean_pooling_convolve(output_feature_maps_correlate)
    end_time = time.clock()
    output_feature_maps_mean_convolve = pooling_layer.out_fea_maps
    print('elapsed time for mean pooling convolve method:... %f' % (end_time - start_time))

    start_time = time.clock()
    pooling_layer.calc_max_pooling_loop(output_feature_maps_correlate)
    end_time = time.clock()
    output_feature_maps_max_loop = pooling_layer.out_fea_maps
    print('elapsed time for max pooling loop method:... %f' % (end_time - start_time))

    start_time = time.clock()
    pooling_layer.calc_max_pooling_split(output_feature_maps_correlate)
    end_time = time.clock()
    output_feature_maps_max_split = pooling_layer.out_fea_maps
    print('elapsed time for max pooling split method:... %f' % (end_time - start_time))

    # testify
    print(np.allclose(output_feature_maps_mean_loop, output_feature_maps_mean_convolve))
    print(np.allclose(output_feature_maps_max_loop, output_feature_maps_max_split))

if __name__ == '__main__':
    test_pooling()
