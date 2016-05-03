
import numpy as np
from scipy import signal

_floatX = np.float32

class ConvolveLayer(object):

    def __init__(self, filter_shape, in_feature_shape, act_func):

        fan_in = filter_shape[0] * filter_shape[2] * filter_shape[3]
        fan_out = filter_shape[1] * filter_shape[2] * filter_shape[3]
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        self.W = np.asarray(
            np.random.uniform(
                low=-w_bound,
                high=w_bound,
                size=filter_shape
            ),
            dtype=_floatX
        )
        self.b = np.zeros((filter_shape[1],), dtype=_floatX)

        self.act_func = act_func

        if 'sigmoid' == self.act_func:
            self.W *= 4
            print('convolutional layer activation function is sigmoid')
        elif 'tanh' == self.act_func:
            print('convolutional layer activation function is tanh')
        elif 'relu' == self.act_func:
            print('convolutional layer activation function is relu')
        else:
            raise NotImplementedError()

        self.grad_W = np.zeros(self.W.shape, dtype=_floatX)
        self.grad_b = np.zeros(self.b.shape, dtype=_floatX)
        self.n_feat_maps, self.n_filters, self.n_filter_h, self.n_filter_w = filter_shape
        self.n_feat_h, self.n_feat_w = in_feature_shape[1:]
        self.n_o_feat_h = self.n_feat_h - self.n_filter_h + 1
        self.n_o_feat_w = self.n_feat_w - self.n_filter_w + 1
        self.o_feat_shape = [self.n_filters, self.n_o_feat_h, self.n_o_feat_w]
        self.n_out = np.prod(self.o_feat_shape[:])

    def sigmoid(self, x, min_val=-30, max_val=30):
        x.clip(min=min_val, max=max_val)
        return 1.0 / (1.0 + np.exp(-x))

    def relu(self, x):
        return x * (x > 0)

    def relu_grad(self, x):
        return 1. * (x > 0)

    def calc_convolve_correlate(self, in_feat_maps):
        n_batch_size = in_feat_maps.shape[0]
        self.o_feat_maps = np.zeros(([n_batch_size, self.n_filters, self.n_o_feat_h, self.n_o_feat_w]), dtype=_floatX)
        for imgIdx in range(n_batch_size):
            for filterIdx in range(self.n_filters):
                convolved_image = np.zeros([self.n_o_feat_h, self.n_o_feat_w])
                for inFeatIdx in range(self.n_feat_maps):
                    filter = self.W[inFeatIdx, filterIdx, :, :]
                    im = in_feat_maps[imgIdx, inFeatIdx, :, :]
                    convolved_image += signal.correlate2d(im, filter, 'valid')

                convolved_image += self.b[filterIdx]
                if 'sigmoid' == self.act_func:
                    convolved_image = self.sigmoid(convolved_image)
                elif 'tanh' == self.act_func:
                    convolved_image = np.tanh(convolved_image)
                elif 'relu' == self.act_func:
                    convolved_image = self.relu(convolved_image)
                else:
                    raise NotImplementedError()
                self.o_feat_maps[imgIdx, filterIdx, :, :] = convolved_image
        self.o_feat_maps_v = self.o_feat_maps.reshape(n_batch_size, self.n_out)

    def calc_delta_p(self, next_layer_delta):
        n_batch_size = next_layer_delta.shape[0]
        self.delta = np.zeros(([n_batch_size, self.n_filters, self.n_o_feat_h, self.n_o_feat_w]), dtype=_floatX)

        if 'sigmoid' == self.act_func:
            self.delta = next_layer_delta * (self.o_feat_maps - self.o_feat_maps ** 2)
        elif 'tanh' == self.act_func:
            self.delta = next_layer_delta * (1 - self.o_feat_maps ** 2)
        elif 'relu' == self.act_func:
            self.delta = next_layer_delta * self.relu_grad(self.o_feat_maps)
        else:
            raise NotImplementedError()

    def calc_delta_v(self, next_w, next_delta):

        n_batch_size = next_delta.shape[0]
        if 'sigmoid' == self.act_func:
            self.delta_v = np.dot(next_delta, next_w.transpose()) * (self.o_feat_maps_v - self.o_feat_maps_v ** 2)
        elif 'tanh' == self.act_func:
            self.delta_v = np.dot(next_delta, next_w.transpose()) * (1 - self.o_feat_maps_v ** 2)
        elif 'relu' == self.act_func:
            self.delta_v = np.dot(next_delta, next_w.transpose()) * self.relu_grad(self.o_feat_maps_v)
        else:
            raise NotImplementedError()
        self.delta = self.delta_v.reshape([n_batch_size, self.n_filters, self.n_o_feat_h, self.n_o_feat_w])

    def calc_delta_c(self, next_w, next_delta):

        n_batch_size = next_delta.shape[0]
        self.delta = np.zeros([n_batch_size, self.n_filters, self.n_o_feat_h, self.n_o_feat_w])
        n_next_filters = next_w.shape[1]

        for imgIdx in range(n_batch_size):
            for featIdx in range(self.n_filters):
                temp_delta = np.zeros([self.n_o_feat_h, self.n_o_feat_w])
                for filterIdx in range(n_next_filters):
                    temp_delta += signal.convolve2d(next_delta[imgIdx, filterIdx, :, :], next_w[featIdx, filterIdx, :, :], 'full')
                self.delta[imgIdx, featIdx, :, :] = temp_delta

    def calc_grad(self, in_feat_maps):
        n_batch_size = in_feat_maps.shape[0]
        for filterIdx in range(self.n_filters):
            for inFeatIdx in range(self.n_feat_maps):
                temp_w_grad = np.zeros([self.n_filter_h, self.n_filter_w])
                for imgIdx in range(n_batch_size):
                    error_matrix = self.delta[imgIdx, filterIdx, :, :]
                    im = in_feat_maps[imgIdx, inFeatIdx, :, :]
                    temp_w_grad += signal.correlate2d(im, error_matrix, 'valid')
                self.grad_W[inFeatIdx, filterIdx, :, :] = -temp_w_grad * (1 / n_batch_size)
            self.grad_b[filterIdx] = -(self.delta[:, filterIdx, :, :].sum()) * (1 / n_batch_size)

    def update_params(self, learning_rate, reg_param):
        self.W -= learning_rate * (self.grad_W + reg_param * self.W)
        self.b -= learning_rate * self.grad_b

    def calc_reg_cost(self, reg_param):
        reg_cost = reg_param / 2 * (self.W ** 2).sum()
        return reg_cost
