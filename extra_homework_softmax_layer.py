
import numpy as np
import timeit
import os
from extra_homework_utils import *


_floatX = np.float32
_intX = np.int16


class SoftmaxLayer(object):

    def __init__(self, n_in, n_out):
        self.n_in = n_in
        self.n_out = n_out
        self.W = np.zeros(shape=[n_in, n_out], dtype=_floatX)
        self.b = np.zeros(shape=[n_out, ], dtype=_floatX)

    # prevent numerical problem
    def safe_ln(self, x, min_val=1e-10):
        return np.log(x.clip(min=min_val))

    # forward for softmax
    # x is a mini-batch, whose dimension is [#sample, #feature]
    # each row is a sample, each column is a feature
    def calc_forward(self, x):

        w_x_b = np.dot(x, self.W) + self.b
        # prevent overflow
        row_max = np.max(w_x_b, axis=1)
        w_x_b -= row_max.reshape(row_max.shape[0], 1)
        # softmax function
        exp_w_x_b = np.exp(w_x_b)
        row_sum = np.sum(exp_w_x_b, axis=1)
        self.py_given_x = exp_w_x_b / row_sum.reshape(row_sum.shape[0], 1)

    # x is a mini-batch and y is the corresponding column vector for labels
    def calc_delta_and_grad(self, x, y):
        # calc delta
        n_batch_size = y.shape[0]
        ground_truth = np.zeros((y.shape[0], self.n_out), dtype=_intX)
        ground_truth[np.arange(y.shape[0]), y] = 1
        self.delta = ground_truth - self.py_given_x
        # calc gradient
        self.grad_w = -np.dot(x.transpose(), self.delta) / n_batch_size
        self.grad_b = -np.mean(self.delta, axis=0)

    # apply weight decay here
    def update_params(self, learning_rate, reg_param):
        self.W -= learning_rate * (self.grad_w + reg_param * self.W)
        self.b -= learning_rate * self.grad_b

    # calculate weight decay cost
    def calc_reg_cost(self, reg_param):
        reg_cost = reg_param / 2 * (self.W ** 2).sum()
        return reg_cost

    # backward for softmax
    def calc_backward(self, x, y, learning_rate, reg_param):
        self.calc_delta_and_grad(x, y)
        self.update_params(learning_rate, reg_param)

    # calc negative log likelihood cost
    def negative_log_likelihood(self, y):
        log_prob = self.safe_ln(self.py_given_x)
        nll_cost = -np.mean(log_prob[np.arange(y.shape[0]), y])
        return nll_cost

    def overall_cost(self, y, reg_param):
        reg_cost = self.calc_reg_cost(reg_param)
        nll_cost = self.negative_log_likelihood(y)
        return nll_cost + reg_cost

    def errors(self, y):
        y_predict = np.argmax(self.py_given_x, axis=1)
        if y.ndim != y_predict.ndim:
            raise TypeError(
                'y should have the same shape as self.y_predict',
                ('y', y.shape, 'y_predict', y_predict.shape)
            )
        if str(y.dtype).startswith('int'):
            return np.mean(y != y_predict)
        else:
            raise NotImplementedError()

def test_softmax():


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
    learning_rate = 0.13
    reg_param = 0.0000
    batch_size = 600
    n_epochs = 1000

    # compute number of mini-batches for training, validation and testing
    n_train_batches = int(np.ceil(train_set_x.shape[0] / batch_size))
    n_valid_batches = int(np.ceil(valid_set_x.shape[0] / batch_size))
    n_test_batches = int(np.ceil(test_set_x.shape[0] / batch_size))

    classifier = SoftmaxLayer(n_in=28*28, n_out=10)

    def test_model(index):
        x = test_set_x[index * batch_size:(index + 1) * batch_size]
        y = test_set_y[index * batch_size:(index + 1) * batch_size]
        classifier.calc_forward(x)
        return classifier.errors(y)

    def validate_model(index):
        x = valid_set_x[index * batch_size:(index + 1) * batch_size]
        y = valid_set_y[index * batch_size:(index + 1) * batch_size]
        classifier.calc_forward(x)
        return classifier.errors(y)

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
                    with open('best_model_softmax.pkl.gz', 'wb') as f:
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
    test_softmax()
