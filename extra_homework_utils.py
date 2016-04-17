
import pickle
import gzip

def load_data(dataset):

    print('...loading data')

    fr = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = pickle.load(fr)
    fr.close()

    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval