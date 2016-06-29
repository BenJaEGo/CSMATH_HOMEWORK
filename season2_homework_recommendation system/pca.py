
import numpy as np
from sklearn.decomposition import PCA

from data_utils import load_movielens_data_from_pickle_split


def calc_RMSE(indicator, rating, predict):
    n_rating = np.sum(indicator.ravel())
    mean_square_error = indicator * ((rating - predict) ** 2) / n_rating
    root_mean_square_error = np.sqrt(np.sum(mean_square_error.ravel()))
    return root_mean_square_error


if __name__ == '__main__':

    # for ml-1m
    # split ratio of training set and test set
    file_dir = 'ml-1m'
    split_ratio = 0.2
    train_data, test_data = load_movielens_data_from_pickle_split(file_dir, split_ratio)
    train_rating, train_indicator, train_timestamp = train_data
    test_rating, test_indicator, test_timestamp = test_data

    pca = PCA(n_components=5)
    pca.fit(train_rating)
    test_projection = pca.transform(test_rating)
    test_origin = pca.inverse_transform(test_projection)

    error = calc_RMSE(test_indicator, test_rating, test_origin)
    print('final RMSE error is %f' % error)
