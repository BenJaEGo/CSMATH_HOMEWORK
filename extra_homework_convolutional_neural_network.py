import numpy as np

from extra_homework_hidden_layer import *
from extra_homework_softmax_layer import *
from extra_homework_convolve_layer import *
from extra_homework_pooling_layer import *
from extra_homework_utils import *


class ConvolutionalNeuralNetwork(object):

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out

        in_feature_shape1 = (1, 28, 28)
        filter_shape1 = (1, 10, 5, 5)
        self.convolve_layer1 = ConvolveLayer(filter_shape1, in_feature_shape1, 'relu')

        pool_shape1 = [2, 2]
        self.pooling_layer1 = PoolingLayer(pool_shape1, self.convolve_layer1.o_feat_shape)

        filter_shape2 = (self.pooling_layer1.n_feature_maps, 20, 5, 5)
        self.convolve_layer2 = ConvolveLayer(filter_shape2, self.pooling_layer1.o_feat_shape, 'relu')

        pool_shape2 = [2, 2]
        self.pooling_layer2 = PoolingLayer(pool_shape2, self.convolve_layer2.o_feat_shape)

        self.hidden_layer = HiddenLayer(self.pooling_layer2.n_out, 500, 'relu')
        self.softmax_layer = SoftmaxLayer(self.hidden_layer.n_out, n_out)


    def forward_propagation(self, input_feature_maps):

        self.convolve_layer1.calc_convolve_correlate(input_feature_maps)
        self.pooling_layer1.calc_max_pooling(self.convolve_layer1.o_feat_maps)

        self.convolve_layer2.calc_convolve_correlate(self.pooling_layer1.o_feat_maps)
        self.pooling_layer2.calc_max_pooling(self.convolve_layer2.o_feat_maps)

        self.hidden_layer.calc_forward(self.pooling_layer2.o_feat_maps_v)

        self.softmax_layer.calc_forward(self.hidden_layer.activation)

    def backward_propagation(self, input_feature_maps, y, learning_rate, reg_param):

        self.softmax_layer.calc_delta_and_grad(self.hidden_layer.activation, y)
        self.hidden_layer.calc_delta_and_grad(self.pooling_layer2.o_feat_maps_v, self.softmax_layer.W, self.softmax_layer.delta)

        self.pooling_layer2.calc_delta_max_v(self.hidden_layer.W, self.hidden_layer.delta)
        self.convolve_layer2.calc_delta_p(self.pooling_layer2.delta_c)
        self.convolve_layer2.calc_grad(self.pooling_layer1.o_feat_maps)

        self.pooling_layer1.calc_delta_max_c(self.convolve_layer2.W, self.convolve_layer2.delta)
        self.convolve_layer1.calc_delta_p(self.pooling_layer1.delta_c)
        self.convolve_layer1.calc_grad(input_feature_maps)

        self.softmax_layer.update_params(learning_rate, reg_param)
        self.hidden_layer.update_params(learning_rate, reg_param)
        self.convolve_layer2.update_params(learning_rate, reg_param)
        self.convolve_layer1.update_params(learning_rate, reg_param)

    def errors(self, x, y):
        self.forward_propagation(x)
        return self.softmax_layer.errors(y)

    def overall_cost(self, y):
        nll_cost = self.softmax_layer.negative_log_likelihood(y)
        # TODO()
        # add weight decay cost for each layer to calculate the overall cost
        return nll_cost


def test_cnn():


    ######################
    #     LOAD DATA      #
    ######################
    dataset = 'MNIST_DATASET.pkl.gz'
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # train_set_x = train_set_x[0:5000, :]
    # train_set_y = train_set_y[0:5000]
    # valid_set_x = valid_set_x[0:2000, :]
    # valid_set_y = valid_set_y[0:2000]
    # test_set_x = test_set_x[0:1000]
    # test_set_y = test_set_y[0:1000]


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('...building the model')

    batch_size = 500
    n_epochs = 200
    learning_rate = 0.1
    reg_param = 0.0000
    height = 28
    width = 28
    n_in = height * width
    n_out = 10

    n_train_batches = int(np.ceil(train_set_x.shape[0] / batch_size))
    n_valid_batches = int(np.ceil(valid_set_x.shape[0] / batch_size))
    n_test_batches = int(np.ceil(test_set_x.shape[0] / batch_size))

    classifier = ConvolutionalNeuralNetwork(n_in, n_out)

    def test_model(index):
        x = test_set_x[index * batch_size:(index + 1) * batch_size]
        y = test_set_y[index * batch_size:(index + 1) * batch_size]

        x = x.reshape(x.shape[0], 1, height, width)

        return classifier.errors(x, y)

    def validate_model(index):
        x = valid_set_x[index * batch_size:(index + 1) * batch_size]
        y = valid_set_y[index * batch_size:(index + 1) * batch_size]

        x = x.reshape(x.shape[0], 1, height, width)

        return classifier.errors(x, y)

    def train_model(index):
        x = train_set_x[index * batch_size:(index + 1) * batch_size]
        y = train_set_y[index * batch_size:(index + 1) * batch_size]
        x = x.reshape(x.shape[0], 1, height, width)

        classifier.forward_propagation(x)
        classifier.backward_propagation(x, y, learning_rate, reg_param)
        return classifier.overall_cost(y)

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    # early-stopping parameters
    # look as this many examples regardless
    patience = 10000
    # wait this much longer when a new best is found
    patience_increase = 2
    # a relative improvement of this much is considered significant
    improvement_threshold = 0.995
    # go through this many mini-batches before checking the network on the validation set
    # in this case we check every epoch
    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses) * 100
#                print(this_validation_loss)

                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches, this_validation_loss))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    # improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    # test it on the test set

                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses) * 100

                    print('epoch %i, minibatch %i/%i, test error of best model %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches, test_score))

                    # save the best model
                    with open('best_model_cnn.pkl.gz', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete. Best validation score of %f %% obtained at iteration %i, with test performance %f %%'
          % (best_validation_loss, best_iter + 1, test_score))
    print('The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time)))
    print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

if __name__ == '__main__':
    test_cnn()
