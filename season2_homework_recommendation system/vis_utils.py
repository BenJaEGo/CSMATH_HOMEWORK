
import numpy as np
import matplotlib.pyplot as plt

_floatX = np.float32
_intX = np.int16


def plot_single_method(training_error_list, test_error_list):
    training_error_list = np.asarray(training_error_list, dtype=_floatX)
    # print(training_error_list.shape)
    test_error_list = np.asarray(test_error_list, dtype=_floatX)
    plt.title('collaborative filtering')
    plt.xlabel('#iter')
    plt.ylabel('RMSE')
    plt.plot(training_error_list[:, 0], training_error_list[:, 1], 'b-', label='training')
    plt.plot(test_error_list[:, 0], test_error_list[:, 1], 'r--', label='test')
    plt.legend()
    plt.show()


def plot_multiple_method(test_error_list):

    plt.title('collaborative filtering')
    plt.xlabel('#iter')
    plt.ylabel('RMSE')

    test_error = np.asarray(test_error_list[0], dtype=_floatX)
    plt.plot(test_error[:, 0], test_error[:, 1], 'b-', label='batch learning vanilla')
    test_error = np.asarray(test_error_list[1], dtype=_floatX)
    plt.plot(test_error[:, 0], test_error[:, 1], 'b--', label='batch learning momentum')
    test_error = np.asarray(test_error_list[2], dtype=_floatX)
    plt.plot(test_error[:, 0], test_error[:, 1], 'r-', label='incomplete incremental learning vanilla')
    test_error = np.asarray(test_error_list[3], dtype=_floatX)
    plt.plot(test_error[:, 0], test_error[:, 1], 'r--', label='incomplete incremental learning momentum')
    test_error = np.asarray(test_error_list[4], dtype=_floatX)
    plt.plot(test_error[:, 0], test_error[:, 1], 'g-', label='complete incremental learning vanilla')
    test_error = np.asarray(test_error_list[5], dtype=_floatX)
    plt.plot(test_error[:, 0], test_error[:, 1], 'g--', label='complete incremental learning momentum')

    plt.legend()
    plt.show()


