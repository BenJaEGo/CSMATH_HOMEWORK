
import numpy as np

from extra_homework_hidden_layer import *
from extra_homework_softmax_layer import *
from extra_homework_utils import *

import timeit
import os


class MultiLayerPerceptron(object):

    def __init__(self, n_in, hidden_layer_params, n_out, act_func):
        self.act_func = act_func
        self.n_in = n_in
        self.n_out = n_out
        self.hidden_layer_params = hidden_layer_params

        self.hidden_layer_list = list()

        if len(self.hidden_layer_params) > 0:
            self.hidden_layer_list.append(HiddenLayer(n_in, hidden_layer_params[0], act_func))

            for i in range(len(hidden_layer_params) - 1):
                self.hidden_layer_list.append(HiddenLayer(hidden_layer_params[i], hidden_layer_params[i+1], act_func))

            self.output_layer = SoftmaxLayer(hidden_layer_params[-1], n_out)
        else:
            self.output_layer = SoftmaxLayer(n_in, n_out)

    def calc_forward(self, x):
        activation = x
        for hidden_layer in self.hidden_layer_list:
            hidden_layer.calc_forward(activation)
            activation = hidden_layer.activation
        self.output_layer.calc_forward(activation)

    def calc_backward(self, x, y, learning_rate, reg_param):

        layer_idx = len(self.hidden_layer_list)
        if layer_idx > 0:
            activation_pre = self.hidden_layer_list[-1].activation
            self.output_layer.calc_delta_and_grad(activation_pre, y)
            next_layer_w = self.output_layer.W
            next_layer_delta = self.output_layer.delta
            self.output_layer.update_params(learning_rate, reg_param)

            while layer_idx > 0:

                if layer_idx > 1:
                    activation_pre = self.hidden_layer_list[layer_idx - 2].activation
                else:
                    activation_pre = x

                self.hidden_layer_list[layer_idx - 1].calc_delta_and_grad(activation_pre, next_layer_w, next_layer_delta)
                next_layer_w = self.hidden_layer_list[layer_idx - 1].W
                next_layer_delta = self.hidden_layer_list[layer_idx - 1].delta

                self.hidden_layer_list[layer_idx - 1].update_params(learning_rate, reg_param)
                layer_idx -= 1
        else:
            self.output_layer.calc_backward(x, y, learning_rate, reg_param)

    def overall_cost(self, y, reg_param):
        i = len(self.hidden_layer_list)
        hidden_layer_cost = 0
        if i > 0:
            while i > 0:
                layer = self.hidden_layer_list[i - 1]
                hidden_layer_cost += layer.calc_reg_cost(reg_param)
                i -= 1

        output_layer_cost = self.output_layer.overall_cost(y, reg_param)
        cost = output_layer_cost + hidden_layer_cost
        return cost

    def errors(self, x, y):
        self.calc_forward(x)
        return self.output_layer.errors(y)


def test_mlp():


    ######################
    #     LOAD DATA      #
    ######################
    dataset = 'MNIST_DATASET.pkl.gz'
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('...building the model')

    batch_size = 20
    n_epochs = 1000

    n_in = 28*28
    n_out = 10
    hidden_layer_params = [10]
    act_func = 'sigmoid'
    learning_rate = 0.01
    reg_param = 0.0001

    # compute number of mini-batches for training, validation and testing
    n_train_batches = int(np.ceil(train_set_x.shape[0] / batch_size))
    # print(n_train_batches)
    n_valid_batches = int(np.ceil(valid_set_x.shape[0] / batch_size))
    n_test_batches = int(np.ceil(test_set_x.shape[0] / batch_size))

    # multi-layer perceptron classifier
    classifier = MultiLayerPerceptron(n_in, hidden_layer_params, n_out, act_func)

    def test_model(index):
        x = test_set_x[index * batch_size:(index + 1) * batch_size]
        y = test_set_y[index * batch_size:(index + 1) * batch_size]

        return classifier.errors(x, y)

    def validate_model(index):
        x = valid_set_x[index * batch_size:(index + 1) * batch_size]
        y = valid_set_y[index * batch_size:(index + 1) * batch_size]

        return classifier.errors(x, y)

    def train_model(index):
        x = train_set_x[index * batch_size:(index + 1) * batch_size]
        y = train_set_y[index * batch_size:(index + 1) * batch_size]

        classifier.calc_forward(x)
        classifier.calc_backward(x, y, learning_rate, reg_param)
        return classifier.overall_cost(y, reg_param)


    ###############
    # TRAIN MODEL #
    ###############
    print('...training the model')
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
    test_score = 0
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    best_iter = 0
    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)

            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses) * 100

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
                    with open('best_model_mlp.pkl.gz', 'wb') as f:
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
    test_mlp()
