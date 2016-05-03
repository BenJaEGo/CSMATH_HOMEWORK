
import numpy as np
from scipy import signal
from extra_homework_utils import *

import time


class ConvolveLayer(object):

    # W [featureMapsNumbers, filterNumber, filterHeight, filterWidth]
    # input_feature_maps [mini_batch_size, featureMapsNumbers, featureHeight, featureWidth]
    # in_feature_shape [featureMapsNumbers, featureHeight, featureWidth]
    # filter_shape [featureMapsNumbers, filterNumber, filterHeight, filterWidth]

    def __init__(self, filter_shape, in_feature_shape, act_func):

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[0] * filter_shape[2:])

        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width"
        fan_out = np.prod(filter_shape[1:])

        w_bound = np.sqrt(6. / (fan_in + fan_out))

        # initialize weights with random weights
        self.W = np.asarray(
            np.random.uniform(
                low=-1.0/w_bound,
                high=1.0/w_bound,
                size=filter_shape
            ),
            dtype=np.float64
        )

        # the bias is 1D -- one bias per output feature map
        self.b = np.zeros(shape=[filter_shape[1], ], dtype=np.float64)

        self.act_func = act_func

        self.reg_cost = 0

        self.grad_W = np.zeros(self.W.shape, dtype=np.float64)
        self.grad_b = np.zeros(self.b.shape, dtype=np.float64)
        self.n_fea_maps, self.n_filters, self.n_filter_h, self.n_filter_w = filter_shape
        self.n_fea_h, self.n_fea_w = in_feature_shape[1:]
        self.n_out_fea_h = self.n_fea_h - self.n_filter_h + 1
        self.n_out_fea_w = self.n_fea_w - self.n_filter_w + 1
        self.out_fea_shape = [self.n_filters, self.n_out_fea_h, self.n_out_fea_w]
        self.n_out = np.prod(self.out_fea_shape[:])

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def relu(self, x):
        return x * (x > 0)

    def relu_grad(self, x):
        return 1. * (x > 0)

    def calc_convolve_loop(self, input_feature_maps):

        n_batch_size = input_feature_maps.shape[0]
        self.out_fea_maps = np.zeros([n_batch_size, self.n_filters, self.n_out_fea_h, self.n_out_fea_w])
        for imgInd in range(n_batch_size):
            for filtInd in range(self.n_filters):
                convolved_image = np.zeros([self.n_out_fea_h, self.n_out_fea_w])
                for inFeaInd in range(self.n_fea_maps):
                    current_filter = self.W[inFeaInd, filtInd, :, :]
                    im = input_feature_maps[imgInd, inFeaInd, :, :]
                    for hInd in range(self.n_out_fea_h):
                        for wInd in range(self.n_out_fea_w):
                            im_patch_height = hInd + self.n_filter_h
                            im_patch_width = wInd + self.n_filter_w
                            im_patch = im[hInd:im_patch_height, wInd:im_patch_width]
                            convolved_image[hInd, wInd] += (current_filter * im_patch).sum()
                convolved_image += self.b[filtInd]
                if 'sigmoid' == self.act_func:
                    convolved_image = self.sigmoid(convolved_image)
                elif 'tanh' == self.act_func:
                    convolved_image = np.tanh(convolved_image)
                elif 'rectified' == self.act_func:
                    convolved_image = self.relu(convolved_image)
                else:
                    raise NotImplementedError()
                self.out_fea_maps[imgInd, filtInd, :, :] = convolved_image
        self.out_fea_maps_v = self.out_fea_maps.reshape(n_batch_size, self.n_out)

    def calc_convolve_convolve(self, input_feature_maps):

        n_batch_size = input_feature_maps.shape[0]
        self.out_fea_maps = np.zeros([n_batch_size, self.n_filters, self.n_out_fea_h, self.n_out_fea_w])
        for imgInd in range(n_batch_size):
            for filtInd in range(self.n_filters):
                convolved_image = np.zeros([self.n_out_fea_h, self.n_out_fea_w])
                for inFeaInd in range(self.n_fea_maps):
                    current_filter = np.rot90(self.W[inFeaInd, filtInd, :, :], 2)
                    im = input_feature_maps[imgInd, inFeaInd, :, :]
                    convolved_image += signal.convolve2d(im, current_filter, 'valid')

                convolved_image += self.b[filtInd]
                if 'sigmoid' == self.act_func:
                    convolved_image = self.sigmoid(convolved_image)
                elif 'tanh' == self.act_func:
                    convolved_image = np.tanh(convolved_image)
                elif 'rectified' == self.act_func:
                    convolved_image = self.relu(convolved_image)
                else:
                    raise NotImplementedError()
                self.out_fea_maps[imgInd, filtInd, :, :] = convolved_image
        self.out_fea_maps_v = self.out_fea_maps.reshape(n_batch_size, self.n_out)

    def calc_convolve_correlate(self, input_feature_maps):

        n_batch_size = input_feature_maps.shape[0]
        self.out_fea_maps = np.zeros([n_batch_size, self.n_filters, self.n_out_fea_h, self.n_out_fea_w])
        for imgInd in range(n_batch_size):
            for filtInd in range(self.n_filters):
                convolved_image = np.zeros([self.n_out_fea_h, self.n_out_fea_w])
                for inFeaInd in range(self.n_fea_maps):
                    current_filter = self.W[inFeaInd, filtInd, :, :]
                    im = input_feature_maps[imgInd, inFeaInd, :, :]
                    convolved_image += signal.correlate2d(im, current_filter, 'valid')

                convolved_image += self.b[filtInd]
                if 'sigmoid' == self.act_func:
                    convolved_image = self.sigmoid(convolved_image)
                elif 'tanh' == self.act_func:
                    convolved_image = np.tanh(convolved_image)
                elif 'rectified' == self.act_func:
                    convolved_image = self.relu(convolved_image)
                else:
                    raise NotImplementedError()
                self.out_fea_maps[imgInd, filtInd, :, :] = convolved_image
        self.out_fea_maps_v = self.out_fea_maps.reshape(n_batch_size, self.n_out)

    def calc_convolve_fourier_loop(self, input_feature_maps):

        n_batch_size = input_feature_maps.shape[0]
        self.out_fea_maps = np.zeros([n_batch_size, self.n_filters, self.n_out_fea_h, self.n_out_fea_w])
        fft_shape_h = self.n_fea_h + self.n_filter_h - 1
        fft_shape_w = self.n_fea_w + self.n_filter_w - 1
        fft_shape = (fft_shape_h, fft_shape_w)
        for imgInd in range(n_batch_size):
            for filtInd in range(self.n_filters):
                convolved_image = np.zeros([self.n_out_fea_h, self.n_out_fea_w])
                for inFeaInd in range(self.n_fea_maps):
                    current_filter = self.W[inFeaInd, filtInd, :, :]
                    current_filter = np.fft.fft2(np.rot90(current_filter, 2), fft_shape)
                    im = input_feature_maps[imgInd, inFeaInd, :, :]
                    im = np.fft.fft2(im, fft_shape)
                    im_fft = np.real(np.fft.ifft2(im * current_filter))
                    convolved_image += im_fft[self.n_filter_h-1:self.n_fea_h, self.n_filter_w-1:self.n_fea_w]

                convolved_image += self.b[filtInd]
                if 'sigmoid' == self.act_func:
                    convolved_image = self.sigmoid(convolved_image)
                elif 'tanh' == self.act_func:
                    convolved_image = np.tanh(convolved_image)
                elif 'rectified' == self.act_func:
                    convolved_image = self.relu(convolved_image)
                else:
                    raise NotImplementedError()
                self.out_fea_maps[imgInd, filtInd, :, :] = convolved_image
        self.out_fea_maps_v = self.out_fea_maps.reshape(n_batch_size, self.n_out)

    def calc_convolve_fourier_partial_batch(self, input_feature_maps):

        n_batch_size = input_feature_maps.shape[0]
        self.out_fea_maps = np.zeros([n_batch_size, self.n_filters, self.n_out_fea_h, self.n_out_fea_w])
        fft_shape_h = self.n_fea_h + self.n_filter_h - 1
        fft_shape_w = self.n_fea_w + self.n_filter_w - 1
        fft_shape = (fft_shape_h, fft_shape_w)

        # using batch fourier process, but loop over filters
        for filtInd in range(self.n_filters):
            for inFeaInd in range(self.n_fea_maps):
                current_filter = self.W[inFeaInd, filtInd, :, :]
                current_filter = np.fft.fft2(np.rot90(current_filter, 2), fft_shape)
                im = np.fft.fft2(input_feature_maps, fft_shape)
                im_fft = np.real(np.fft.ifft2(im * current_filter))
                im_fft = im_fft.reshape([im_fft.shape[0], im_fft.shape[2], im_fft.shape[3]])
                # print(im_fft.shape)
                self.out_fea_maps[:, filtInd, :, :] = im_fft[:, self.n_filter_h-1:self.n_fea_h, self.n_filter_w-1:self.n_fea_w]
                self.out_fea_maps[:, filtInd, :, :] += self.b[filtInd]
                if 'sigmoid' == self.act_func:
                    self.out_fea_maps[:, filtInd, :, :] = self.sigmoid(self.out_fea_maps[:, filtInd, :, :])
                elif 'tanh' == self.act_func:
                    self.out_fea_maps[:, filtInd, :, :] = np.tanh(self.out_fea_maps[:, filtInd, :, :])
                elif 'rectified' == self.act_func:
                    self.out_fea_maps[:, filtInd, :, :] = self.relu(self.out_fea_maps[:, filtInd, :, :])
                else:
                    raise NotImplementedError()
        self.out_fea_maps_v = self.out_fea_maps.reshape(n_batch_size, self.n_out)


    def calc_convolve_fourier_full_batch(self, input_feature_maps):

        n_batch_size = input_feature_maps.shape[0]
        self.out_fea_maps = np.zeros([n_batch_size, self.n_filters, self.n_out_fea_h, self.n_out_fea_w])
        fft_shape_h = self.n_fea_h + self.n_filter_h - 1
        fft_shape_w = self.n_fea_w + self.n_filter_w - 1
        fft_shape = (fft_shape_h, fft_shape_w)

        # validate the filters
        # rot90 only rotate the first two dimensions
        # so to rotate filters in 3D, must do these following changes
        # after rotate, changes it back
        # filters = self.W.transpose(1, 2, 0).reshape([self.W.shape[1], self.W.shape[2], self.W.shape[0]])
        # filters = np.rot90(filters, 2)
        # filters = filters.transpose(2, 0, 1).reshape([filters.shape[2], filters.shape[0], filters.shape[1]])
        # filters = np.fft.fft2(filters, fft_shape)
        filters = self.W.transpose(2, 3, 0, 1).reshape([self.n_filter_h, self.n_filter_w, self.n_fea_maps, self.n_filters])
        filters = np.rot90(filters, 2)
        # print(filters.shape)
        filters = filters.transpose(2, 3, 0, 1).reshape(self.W.shape)
        # print(filters.shape)
        filters = np.fft.fft2(filters, fft_shape)
        im = np.fft.fft2(input_feature_maps, fft_shape)
        im_fft = np.real(np.fft.ifft2(im * filters))
        self.out_fea_maps = im_fft[:, :, self.n_filter_h - 1:self.n_fea_h, self.n_filter_w - 1:self.n_fea_w]
        for filtInd in range(self.n_filters):
            self.out_fea_maps[:, filtInd, :, :] += self.b[filtInd]
        if 'sigmoid' == self.act_func:
            self.out_fea_maps = self.sigmoid(self.out_fea_maps)
        elif 'tanh' == self.act_func:
            self.out_fea_maps = np.tanh(self.out_fea_maps)
            # print('test')
        elif 'rectified' == self.act_func:
            self.out_fea_maps = self.relu(self.out_fea_maps)
        else:
            raise NotImplementedError()
        self.out_fea_maps_v = self.out_fea_maps.reshape(n_batch_size, self.n_out)

    # convolve layer is followed by a pooling layer
    def calc_delta_p(self, next_layer_delta):
        n_batch_size = next_layer_delta.shape[0]
        self.delta = np.zeros([n_batch_size, self.n_filters, self.n_out_fea_h, self.n_out_fea_w], dtype=np.float64)

        if 'sigmoid' == self.activation:
            self.delta = next_layer_delta * (self.out_fea_maps - self.out_fea_maps ** 2)
        elif 'tanh' == self.activation:
            self.delta = next_layer_delta * (1 - self.out_fea_maps ** 2)
        elif 'rectified' == self.activation:
            self.delta = next_layer_delta * self.rectified_grad(self.out_fea_maps)
        else:
            raise NotImplementedError()

    # in case convolve layer is followed by a normal hidden layer or a softmax layer
    def calc_delta_v(self, next_layer_W, next_layer_delta):
        n_batch_size = next_layer_delta.shape[0]
        self.delta_v = np.zeros((next_layer_delta.shape[0], next_layer_W.shape[0]), dtype=np.float64)
        self.delta = np.zeros([n_batch_size, self.n_filters, self.n_out_fea_h, self.n_out_fea_w], dtype=np.float64)
        self.delta_v = np.dot(next_layer_delta, next_layer_W.transpose())
        self.delta = self.delta_v.reshape([n_batch_size, self.n_filters, self.n_out_fea_h, self.n_out_fea_w])

    # in case convolve layer is followed by a convolve layer
    def calc_delta_c(self, next_layer_W, next_layer_delta):

        n_batch_size = next_layer_delta.shape[0]
        self.delta = np.zeros([n_batch_size, self.n_filters, self.n_out_fea_h, self.n_out_fea_w])
        n_next_filters = next_layer_W.shape[1]

        for imgIdn in range(n_batch_size):
            for feaIdn in range(self.n_filters):
                temp_delta = np.zeros([self.n_out_fea_h, self.n_out_fea_w])
                for filtIdn in range(n_next_filters):
                    temp_delta += signal.convolve2d(next_layer_delta[imgIdn, filtIdn, :, :], next_layer_W[feaIdn, filtIdn, :, :], 'full')
                self.delta[imgIdn, feaIdn, :, :] = temp_delta

    def calc_grad(self, input_feature_maps):

        n_batch_size = input_feature_maps.shape[0]
        for filtInd in range(self.n_filters):
            for inFeaInd in range(self.n_fea_maps):
                temp_w_grad = np.zeros([self.n_filter_h, self.n_filter_w])
                for imgInd in range(n_batch_size):
                    error_matrix = self.delta[imgInd, filtInd, :, :]
                    im = input_feature_maps[imgInd, inFeaInd, :, :]
                    temp_w_grad += signal.correlate2d(im, error_matrix, 'valid')
                self.grad_W[inFeaInd, filtInd, :, :] = -temp_w_grad * (1 / n_batch_size)
            self.grad_b[filtInd] = -(self.delta[:, filtInd, :, :].sum()) * (1 / n_batch_size)

    def update_params(self, learning_rate, L1_reg, L2_reg):
        self.W -= learning_rate * (self.grad_W + L1_reg * self.W + L2_reg * self.W)
        self.b -= learning_rate * self.grad_b

    def calc_weightdecay_cost(self):
        self.L1_reg_cost = abs(self.W).sum()
        self.L2_reg_cost = (self.W ** 2).sum()



def test_conv():
    dataset = 'MNIST_DATASET.pkl.gz'
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    train_data = train_set_x[0:1000]
    test_data = train_set_y[0:1000]

    in_feature_shape = (1, 28, 28)
    filter_shape = (1, 10,  5, 5)

    n_batch_size = train_data.shape[0]
    height = 28
    width = 28

    in_feature_maps = train_data.reshape(n_batch_size, 1, height, width)

    convolve_layer = ConvolveLayer(filter_shape, in_feature_shape, 'rectified')

    start_time = time.clock()
    convolve_layer.calc_convolve_convolve(in_feature_maps)
    end_time = time.clock()
    output_feature_maps_convolve = convolve_layer.out_fea_maps
    print('elapsed time for convolve method:... %f' % (end_time - start_time))

    start_time = time.clock()
    convolve_layer.calc_convolve_correlate(in_feature_maps)
    end_time = time.clock()
    output_feature_maps_correlate = convolve_layer.out_fea_maps
    print('elapsed time for correlate method:... %f' % (end_time - start_time))

    start_time = time.clock()
    convolve_layer.calc_convolve_fourier_full_batch(in_feature_maps)
    end_time = time.clock()
    output_feature_maps_fourier_full = convolve_layer.out_fea_maps
    print('elapsed time for full fourier method:... %f' % (end_time - start_time))

    start_time = time.clock()
    convolve_layer.calc_convolve_fourier_partial_batch(in_feature_maps)
    end_time = time.clock()
    output_feature_maps_fourier_partial = convolve_layer.out_fea_maps
    print('elapsed time for patial fourier method:... %f' % (end_time - start_time))

    start_time = time.clock()
    convolve_layer.calc_convolve_fourier_loop(in_feature_maps)
    end_time = time.clock()
    output_feature_maps_fourier_loop = convolve_layer.out_fea_maps
    print('elapsed time for loop fourier method:... %f' % (end_time - start_time))

    # testify
    print(np.allclose(output_feature_maps_convolve, output_feature_maps_correlate))
    print(np.allclose(output_feature_maps_convolve, output_feature_maps_fourier_full))
    print(np.allclose(output_feature_maps_convolve, output_feature_maps_fourier_loop))
    print(np.allclose(output_feature_maps_convolve, output_feature_maps_fourier_partial))

if __name__ == '__main__':
    test_conv()
