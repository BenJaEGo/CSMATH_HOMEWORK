
import numpy as np

_floatX = np.float32

class HiddenLayer(object):

    def __init__(self, n_in, n_out, act_func):

        self.n_in = n_in
        self.n_out = n_out
        self.act_func = act_func

        w_bound = np.sqrt(6.0 / (self.n_in + self.n_out))
        self.W = np.asarray(
            np.random.uniform(
                low=-w_bound,
                high=w_bound,
                size=(n_in, n_out)
            ),
            dtype=_floatX
        )

        if 'sigmoid' == self.act_func:
            self.W *= 4
            print('hidden layer activation function is sigmoid')
        elif 'tanh' == self.act_func:
            print('hidden layer activation function is tanh')
        elif 'relu' == self.act_func:
            print('hidden layer activation function is relu')
        else:
            raise NotImplementedError()

        self.b = np.zeros(shape=[n_out, ], dtype=_floatX)

    def sigmoid(self, x, min_val=-10, max_val=10):
        x = x.clip(min=min_val, max=max_val)
        return 1.0 / (1.0 + np.exp(-x))

    def relu(self, x):
        return x * (x > 0)

    def relu_grad(self, x):
        return 1. * (x > 0)

    def calc_forward(self, x):
        self.w_x_b = np.dot(x, self.W) + self.b
        if 'sigmoid' == self.act_func:
            self.activation = self.sigmoid(self.w_x_b)
        elif 'tanh' == self.act_func:
            self.activation = np.tanh(self.w_x_b)
        elif 'relu' == self.act_func:
            self.activation = self.relu(self.w_x_b)
        else:
            raise NotImplementedError()

    def calc_delta_and_grad(self, x, next_w, next_delta):
        n_batch_size = next_delta.shape[0]
        # calc delta
        # self.delta = np.zeros(shape=[n_batch_size, self.n_out], dtype=_floatX)
        if 'sigmoid' == self.act_func:
            self.delta = np.dot(next_delta, next_w.transpose()) * (self.activation - self.activation ** 2)
        elif 'tanh' == self.act_func:
            self.delta = np.dot(next_delta, next_w.transpose()) * (1 - self.activation ** 2)
        elif 'relu' == self.act_func:
            self.delta = np.dot(next_delta, next_w.transpose()) * self.relu_grad(self.w_x_b)
        else:
            raise NotImplementedError()

        # calc gradient
        self.grad_w = -np.dot(x.transpose(), self.delta) / n_batch_size
        self.grad_b = -np.mean(self.delta, axis=0)

    def update_params(self, learning_rate, reg_param):
        self.W -= learning_rate * (self.grad_w + reg_param * self.W)
        self.b -= learning_rate * self.grad_b

    def calc_reg_cost(self, reg_param):
        reg_cost = reg_param / 2 * (self.W ** 2).sum()
        return reg_cost
