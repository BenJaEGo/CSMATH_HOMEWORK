
import numpy as np
from scipy import signal
import scipy as sp

_floatX = np.float32

class PoolingLayer(object):

    def __init__(self, pool_shape, input_feature_maps_shape):
        self.pool_shape = pool_shape
        self.pool_shape_h = pool_shape[0]
        self.pool_shape_w = pool_shape[1]

        self.n_feature_maps = input_feature_maps_shape[0]
        self.delta_size_h = input_feature_maps_shape[1]
        self.delta_size_w = input_feature_maps_shape[2]
        self.pooled_h = input_feature_maps_shape[1] // self.pool_shape_h
        self.pooled_w = input_feature_maps_shape[2] // self.pool_shape_w
        self.n_out = self.n_feature_maps * self.pooled_h * self.pooled_w
        self.o_feat_shape = [self.n_feature_maps, self.pooled_h, self.pooled_w]

    def calc_max_pooling(self, in_feat_maps):

        n_batch_size = in_feat_maps.shape[0]
        temp = np.asarray(np.split(in_feat_maps, self.pooled_w, axis=3))
        temp = temp.transpose(1, 2, 3, 0, 4).reshape([n_batch_size, self.n_feature_maps, in_feat_maps.shape[2], self.pooled_w, self.pool_shape_w])
        temp = np.asarray(np.split(temp, self.pooled_h, axis=2))
        temp = temp.transpose(1, 2, 0, 4, 3, 5).reshape([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w, self.pool_shape_h, self.pool_shape_w])
        temp = temp.reshape([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w, np.prod(self.pool_shape[:])])
        self.position = np.argmax(temp, axis=4)
        self.o_feat_maps = np.max(temp, axis=4)
        self.o_feat_maps_v = self.o_feat_maps.reshape(n_batch_size, self.n_out)

    def calc_delta_max_v(self, next_w, next_delta):

        n_batch_size = next_delta.shape[0]
        self.delta_c = np.zeros([n_batch_size, self.n_feature_maps, self.delta_size_h, self.delta_size_w])

        self.delta_v = np.dot(next_delta, next_w.transpose())
        self.delta = self.delta_v.reshape([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w])
        tmp_delta_c = self.delta_c.reshape(np.prod(self.delta.shape[:]), np.prod(self.pool_shape[:]))
        tmp_delta = self.delta.reshape(np.prod(self.delta.shape[:]), )
        tmp_index = self.position.reshape(np.prod(self.position.shape[:]), )
        tmp_delta_c[np.arange(tmp_delta_c.shape[0]), tmp_index] = tmp_delta
        self.delta_c = tmp_delta_c.reshape(self.delta_c.shape)

    def calc_delta_max_c(self, next_w, next_delta):

        n_batch_size = next_delta.shape[0]
        self.delta = np.zeros([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w])
        n_filters = next_w.shape[1]

        for imgIdx in range(n_batch_size):
            for featIdx in range(self.n_feature_maps):
                temp_delta = np.zeros([self.pooled_h, self.pooled_w])
                for filterIdx in range(n_filters):
                    temp_delta += signal.convolve2d(next_delta[imgIdx, filterIdx, :, :], next_w[featIdx, filterIdx, :, :], 'full')
                self.delta[imgIdx, featIdx, :, :] = temp_delta
        self.delta_v = self.delta.reshape([n_batch_size, self.n_out])

        self.delta_c = np.zeros([n_batch_size, self.n_feature_maps, self.delta_size_h, self.delta_size_w])
        tmp_delta_c = self.delta_c.reshape(np.prod(self.delta.shape[:]), np.prod(self.pool_shape[:]))
        tmp_delta = self.delta.reshape(np.prod(self.delta.shape[:]), )
        tmp_index = self.position.reshape(np.prod(self.position.shape[:]), )
        tmp_delta_c[np.arange(tmp_delta_c.shape[0]), tmp_index] = tmp_delta
        self.delta_c = tmp_delta_c.reshape(self.delta_c.shape)

    def calc_mean_pooling(self, in_feat_maps):

        n_batch_size = in_feat_maps.shape[0]
        self.o_feat_maps = np.zeros([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w])
        for imgIdx in range(n_batch_size):
            for filterIdx in range(self.n_feature_maps):
                im = in_feat_maps[imgIdx, filterIdx, :, :]
                p_feat_con = signal.correlate2d(im, np.ones(self.pool_shape), 'valid')
                factor = 1 / (self.pool_shape[0] * self.pool_shape[1])
                p_feat_con = p_feat_con[0::self.pool_shape[0], 0::self.pool_shape[1]]
                self.o_feat_maps[imgIdx, filterIdx, :, :] = factor * p_feat_con
        self.o_feat_maps_v = self.o_feat_maps.reshape(n_batch_size, self.n_out)

    def calc_delta_mean_v(self, next_w, next_delta):

        n_batch_size = next_delta.shape[0]
        self.delta_c = np.zeros([n_batch_size, self.n_feature_maps, self.delta_size_h, self.delta_size_w])

        self.delta_v = np.dot(next_delta, next_w.transpose())
        self.delta = self.delta_v.reshape([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w])
        for imgIdx in range(n_batch_size):
            for filteridx in range(self.n_feature_maps):
                self.delta_c[imgIdx, filteridx, :, :] = (1 / np.prod(self.pool_shape[:])) * sp.linalg.kron(np.ones(self.pool_shape), self.delta[imgIdx, filteridx, :, :])

    def calc_delta_mean_c(self, next_w, next_delta):

        n_batch_size = next_delta.shape[0]
        self.delta = np.zeros([n_batch_size, self.n_feature_maps, self.pooled_h, self.pooled_w])
        n_filters = next_w.shape[1]
        for imgIdx in range(n_batch_size):
            for featIdx in range(self.n_feature_maps):
                temp_delta = np.zeros([self.pooled_h, self.pooled_w])
                for filterIdx in range(n_filters):
                    temp_delta += signal.convolve2d(next_delta[imgIdx, filterIdx, :, :], next_w[featIdx, filterIdx, :, :], 'full')
                self.delta[imgIdx, featIdx, :, :] = temp_delta
        self.delta_v = self.delta.reshape([n_batch_size, self.n_out])

        self.delta_c = np.zeros([n_batch_size, self.n_feature_maps, self.delta_size_h, self.delta_size_w])
        for imgIdx in range(n_batch_size):
            for filterIdx in range(self.n_feature_maps):
                self.delta_c[imgIdx, filterIdx, :, :] = (1 / np.prod(self.pool_shape[:])) * sp.linalg.kron(np.ones(self.pool_shape), self.delta[imgIdx, filterIdx, :, :])
