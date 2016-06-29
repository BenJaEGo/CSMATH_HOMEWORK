
import numpy as np
import timeit
import os

from data_utils import load_movielens_data_from_pickle_split
from data_utils import load_movielens_10m_data_from_dat_split
from vis_utils import plot_single_method
from vis_utils import plot_multiple_method

_floatX = np.float32
_intX = np.int16


class CollaborativeFilteringStandard(object):

    def __init__(self, n_feat, lb, ub, cu, co, r=0.1):
        self.n_feature = n_feat
        self.lb = lb
        self.ub = ub
        self.cu = cu
        self.co = co
        self.r = r

    def init(self, ratings, indicator):
        n_users, n_objects = ratings.shape
        sum_rating = np.sum(ratings.ravel())
        n_rating = np.sum(indicator.ravel())
        mean_rating = sum_rating / n_rating
        constant = np.sqrt((mean_rating - self.lb) / self.n_feature)
        users = np.random.uniform(low=-self.r, high=self.r, size=[self.n_feature, n_users]) + constant
        objects = np.random.uniform(low=-self.r, high=self.r, size=[self.n_feature, n_objects]) + constant

        return users, objects

    def calc_predict_clip(self, users, objects):
        predict = np.dot(users.T, objects)
        predict.clip(min=self.lb, max=self.ub)
        return predict

    def calc_RMSE(self, indicator, rating, predict):
        n_rating = np.sum(indicator.ravel())
        mean_square_error = indicator * ((rating - predict) ** 2) / n_rating
        root_mean_square_error = np.sqrt(np.sum(mean_square_error.ravel()))
        return root_mean_square_error

    def calc_gradient(self, indicator, rating, users, objects, mode=0):
        """
        mode 0: batch learning
        mode 1: incomplete incremental learning
        mode 2: complete incremental learning
        """
        if mode == 0:
            predict = self.calc_predict_clip(users, objects)
            temp = indicator * (rating - predict)
            grad_users = -np.dot(objects, temp.T) + self.cu * users
            grad_objects = -np.dot(users, temp) + self.co * objects
            return grad_users, grad_objects
        elif mode == 1:
            predict = self.calc_predict_clip(users, objects)
            temp = indicator * (rating - predict)
            grad_users = -np.dot(objects, temp.T) + self.cu * users
            n_scores_objects = np.sum(indicator, axis=0)
            grad_objects = -np.dot(users, temp) + self.co * n_scores_objects * objects
            return grad_users, grad_objects
        elif mode == 2:
            predict = self.calc_predict_clip(users, objects)
            temp = indicator * (rating - predict)
            n_scores_users = np.sum(indicator, axis=1)
            grad_users = -np.dot(objects, temp.T) + self.cu * n_scores_users * users
            n_scores_objects = np.sum(indicator, axis=0)
            grad_objects = -np.dot(users, temp) + self.co * n_scores_objects * objects
            return grad_users, grad_objects
        else:
            raise NotImplementedError('there exist no such mode for calculate gradient...')

    def update_gradient(self, users, objects, grad_u, grad_o, lr, m, mode=0, pre_grad_u=None, pre_grad_o=None):
        """
        mode 0: vanilla gradient descent
        mode 1: momentum gradient descent
        """
        if mode == 0:
            users += -lr * grad_u
            objects += -lr * grad_o
            return users, objects
        elif mode == 1:
            if pre_grad_u is None or pre_grad_o is None:
                raise ValueError('pre_grad_users and pre_grad_objects can not be None!..')
            grad_u = m * pre_grad_u - lr * grad_u
            grad_o = m * pre_grad_o - lr * grad_o
            users += grad_u
            objects += grad_o
            return users, objects, grad_u, grad_o
        else:
            raise NotImplementedError('there exist no such mode for update gradient...')

    def optimize(self, tr_rating, tr_indicator, te_rating, te_indicator, lr, m, lm=0, um=0, n_iter=100):
        """
        learning mode 0: batch learning
        learning mode 1: incomplete incremental learning
        learning mode 2: complete incremental learning

        update mode 0: vanilla gradient descent
        update mode 1: momentum gradient descent
        """

        users, objects = self.init(tr_rating, tr_indicator)
        train_error_history = list()
        test_error_history = list()
        if um == 1:
            pre_grad_u = np.zeros(shape=users.shape, dtype=_floatX)
            pre_grad_o = np.zeros(shape=objects.shape, dtype=_floatX)
            for i in range(n_iter):
                grad_u, grad_o = self.calc_gradient(tr_indicator, tr_rating, users, objects, lm)
                users, objects, pre_grad_u, pre_grad_o = self.update_gradient(users, objects, grad_u, grad_o, lr, m,
                                                                              um, pre_grad_u, pre_grad_o)
                predicts = self.calc_predict_clip(users, objects)
                train_error = self.calc_RMSE(tr_indicator, tr_rating, predicts)
                test_error = self.calc_RMSE(te_indicator, te_rating, predicts)
                train_error_history.append([i, train_error])
                test_error_history.append([i, test_error])
                print('iter %d, training error %f, test error %f' % (i, train_error, test_error))
            return train_error_history, test_error_history
        elif um == 0:
            for i in range(n_iter):
                grad_u, grad_o = self.calc_gradient(tr_indicator, tr_rating, users, objects, lm)
                users, objects = self.update_gradient(users, objects, grad_u, grad_o, lr, m, um)
                predicts = self.calc_predict_clip(users, objects)
                train_error = self.calc_RMSE(tr_indicator, tr_rating, predicts)
                test_error = self.calc_RMSE(te_indicator, te_rating, predicts)
                train_error_history.append([i, train_error])
                test_error_history.append([i, test_error])
                print('iter %d, training error %f, test error %f' % (i, train_error, test_error))
            return train_error_history, test_error_history
        else:
            raise NotImplementedError('not such update mode..')

if __name__ == '__main__':

    # for ml-1m
    # split ratio of training set and test set
    # file_dir = 'ml-1m'
    # split_ratio = 0.2
    # train_data, test_data = load_movielens_data_from_pickle_split(file_dir, split_ratio)
    # train_rating, train_indicator, train_timestamp = train_data
    # test_rating, test_indicator, test_timestamp = test_data

    # for ml-10m
    file_dir = 'ml-10m'
    split_ratio = 0.2
    train_data, test_data = load_movielens_10m_data_from_dat_split(file_dir, split_ratio)
    train_rating, train_indicator = train_data
    test_rating, test_indicator = test_data

    n_train_rating = int(np.sum(train_indicator.ravel()))
    n_test_rating = int(np.sum(test_indicator.ravel()))
    print('#training rating is %d, #test rating is %d' % (n_train_rating, n_test_rating))
    print('#rating is %d, #test / #train ratio is %f' % (n_train_rating+n_test_rating, n_test_rating/n_train_rating))

    # fixed parameters for movielens dataset
    lower_bound = 1
    upper_bound = 5
    # latent factor number
    n_feature = 10
    # regularized parameters for user features and object features
    co_users = 0.1
    co_objects = 0.1

    cf_obj = CollaborativeFilteringStandard(n_feature, lower_bound, upper_bound, co_users, co_objects)

    # learning rate and momentum for gradient descent
    learning_rate = 0.00001
    momentum = 0.9
    # max iteration number
    max_iter = 50

    # learning_mode can be:
    # 0 batch learning
    # 1 incomplete incremental learning
    # 2 complete incremental learning

    # update_mode can be:
    # 0 vanilla gradient descent
    # 1 momentum gradient descent

    # learning_mode = 0
    # update_mode = 0
    # start_time = timeit.default_timer()
    # train_error_list, test_error_list = cf_obj.optimize(train_rating, train_indicator, test_rating, test_indicator, learning_rate, momentum, learning_mode, update_mode, max_iter)
    # end_time = timeit.default_timer()
    # print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))
    # plot_single_method(train_error_list, test_error_list)

    start_time = timeit.default_timer()
    train_error_00, test_error_00 = cf_obj.optimize(train_rating, train_indicator, test_rating, test_indicator,
                                                    learning_rate, momentum, 0, 0, max_iter)
    train_error_01, test_error_01 = cf_obj.optimize(train_rating, train_indicator, test_rating, test_indicator,
                                                    learning_rate, momentum, 0, 1, max_iter)
    train_error_10, test_error_10 = cf_obj.optimize(train_rating, train_indicator, test_rating, test_indicator,
                                                    learning_rate, momentum, 1, 0, max_iter)
    train_error_11, test_error_11 = cf_obj.optimize(train_rating, train_indicator, test_rating, test_indicator,
                                                    learning_rate, momentum, 1, 1, max_iter)
    train_error_20, test_error_20 = cf_obj.optimize(train_rating, train_indicator, test_rating, test_indicator,
                                                    learning_rate, momentum, 2, 0, max_iter)
    train_error_21, test_error_21 = cf_obj.optimize(train_rating, train_indicator, test_rating, test_indicator,
                                                    learning_rate, momentum, 2, 1, max_iter)

    end_time = timeit.default_timer()
    print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.1fs' % (end_time - start_time)))

    test_error = list()
    test_error.append(test_error_00)
    test_error.append(test_error_01)
    test_error.append(test_error_10)
    test_error.append(test_error_11)
    test_error.append(test_error_20)
    test_error.append(test_error_21)
    plot_multiple_method(test_error)
