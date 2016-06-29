
import numpy as np
import pickle
import gzip
import os
import random

_floatX = np.float32
_intX = np.int8


def load_movielens_data_from_dat(file_dir):

    n_user = 6040
    n_movie = 3952

    rating_matrix = np.zeros(shape=[n_user, n_movie], dtype=_intX)
    indicator_matrix = np.zeros(shape=[n_user, n_movie], dtype=_intX)
    timestamp_matrix = np.zeros(shape=[n_user, n_movie], dtype=_intX)

    ratings_filename = file_dir + r'\ratings.dat'

    with open(ratings_filename, 'r') as f:
        for line in f:
            line = str(line)
            line_split = line.split('::')
            rating_info = np.asarray(line_split, dtype=_intX)

            user_id = rating_info[0]
            movie_id = rating_info[1]
            rating = rating_info[2]
            timestamp = rating_info[3]

            rating_matrix[user_id - 1, movie_id - 1] = rating
            indicator_matrix[user_id-1, movie_id-1] = 1
            timestamp_matrix[user_id-1, movie_id-1] = timestamp

    return rating_matrix, indicator_matrix, timestamp_matrix


def load_movielens_data_from_pickle(file_dir):

    pickle_filename = '%s.pkl.gz' % file_dir

    if os.path.exists(pickle_filename):
        print('pickle file already exists...')
        pass
    else:
        print('pickle file does not exist, initialing...')
        data = load_movielens_data_from_dat(file_dir)
        with gzip.open(pickle_filename, 'wb') as f:
            pickle.dump(data, f)

    print('loading %s data...' % file_dir)

    with gzip.open(pickle_filename, 'rb') as f:
        data = pickle.load(f)
    rating_matrix, indicator_matrix, timestamp_matrix = data
    return rating_matrix, indicator_matrix, timestamp_matrix


def load_movielens_data_from_dat_split(file_dir, split_ratio=0.1):

    n_user = 6040
    n_movie = 3952

    train_rating_matrix = np.zeros(shape=[n_user, n_movie], dtype=_intX)
    train_indicator_matrix = np.zeros(shape=[n_user, n_movie], dtype=_intX)
    train_timestamp_matrix = np.zeros(shape=[n_user, n_movie], dtype=_intX)

    test_rating_matrix = np.zeros(shape=[n_user, n_movie], dtype=_intX)
    test_indicator_matrix = np.zeros(shape=[n_user, n_movie], dtype=_intX)
    test_timestamp_matrix = np.zeros(shape=[n_user, n_movie], dtype=_intX)

    ratings_filename = file_dir + r'\ratings.dat'

    with open(ratings_filename, 'r') as f:
        for line in f:
            line = str(line)
            line_split = line.split('::')
            rating_info = np.asarray(line_split, dtype=_intX)

            user_id = rating_info[0]
            movie_id = rating_info[1]
            rating = rating_info[2]
            timestamp = rating_info[3]

            if random.random() < split_ratio:
                test_rating_matrix[user_id - 1, movie_id - 1] = rating
                test_indicator_matrix[user_id - 1, movie_id - 1] = 1
                test_timestamp_matrix[user_id - 1, movie_id - 1] = timestamp
            else:
                train_rating_matrix[user_id - 1, movie_id - 1] = rating
                train_indicator_matrix[user_id - 1, movie_id - 1] = 1
                train_timestamp_matrix[user_id - 1, movie_id - 1] = timestamp

    train_data = (train_rating_matrix, train_indicator_matrix, train_timestamp_matrix)
    test_data = (test_rating_matrix, test_indicator_matrix, test_timestamp_matrix)

    return train_data, test_data


def load_movielens_data_from_pickle_split(file_dir, split_ratio=0.1):

    pickle_filename = '%s_sp_%.2f.pkl.gz' % (file_dir, split_ratio)

    if os.path.exists(pickle_filename):
        print('pickle file already exists...')
        pass
    else:
        print('pickle file does not exist, initialing...')
        data = load_movielens_data_from_dat_split(file_dir, split_ratio)
        with gzip.open(pickle_filename, 'wb') as f:
            pickle.dump(data, f)

    print('loading %s data...' % file_dir)

    with gzip.open(pickle_filename, 'rb') as f:
        data = pickle.load(f)
    train_data, test_data = data
    return train_data, test_data


def load_movielens_10m_data_from_dat_split(file_dir, split_ratio=0.2):

    # n_user = 71567
    n_movie = 10681

    n_user = 50000

    movies_filename = file_dir + r'\movies.dat'

    movie_ids = list()
    with open(movies_filename, 'r') as f:
        for line in f:
            line = str(line)
            line_split = line.split('::')
            movie_id = int(line_split[0])
            movie_ids.append(movie_id)

    train_rating_matrix = np.zeros(shape=[n_user, n_movie], dtype=_floatX)
    train_indicator_matrix = np.zeros(shape=[n_user, n_movie], dtype=_intX)
    test_rating_matrix = np.zeros(shape=[n_user, n_movie], dtype=_floatX)
    test_indicator_matrix = np.zeros(shape=[n_user, n_movie], dtype=_intX)

    ratings_filename = file_dir + r'\ratings.dat'

    with open(ratings_filename, 'r') as f:
        for line in f:
            line = str(line)
            line_split = line.split('::')
            user_id = int(line_split[0])
            movie_id = int(line_split[1])
            rating = float(line_split[2])

            movie_idx = movie_ids.index(movie_id)
            # if movie_id > 10681:
            #     print(movie_id, movie_idx)
            # print(movie_idx)
            if user_id <= n_user:

                if random.random() < split_ratio:
                    test_rating_matrix[user_id - 1, movie_idx] = rating
                    test_indicator_matrix[user_id-1, movie_idx] = 1

                else:
                    train_rating_matrix[user_id - 1, movie_idx] = rating
                    train_indicator_matrix[user_id - 1, movie_idx] = 1
            else:
                break
    # print(test_rating_matrix.max())
    # print(test_rating_matrix.min())
    # print(train_rating_matrix.max())
    # print(train_rating_matrix.min())
    train_data = (train_rating_matrix, train_indicator_matrix)
    test_data = (test_rating_matrix, test_indicator_matrix)
    return train_data, test_data

if __name__ == '__main__':
    # file_dir = 'ml-1m'
    # split_ratio = 0.2
    # train_data, test_data = load_movielens_data_from_pickle_split(file_dir, split_ratio)

    file_dir = 'ml-10m'
    split_ratio = 0.2
    train_data, test_data = load_movielens_10m_data_from_dat_split(file_dir, split_ratio)
