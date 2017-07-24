'''Imports CIFAR-10 data.'''

import numpy as np
import pickle
import sys
import pprint

pp = pprint.PrettyPrinter(indent=4)

def convertToOneHot(vector, num_classes=None):
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(float)

def load_CIFAR10_batch(filename):
  '''load data from single CIFAR-10 file'''

  with open(filename, 'rb') as f:
    if sys.version_info[0] < 3:
      dict = pickle.load(f)
    else:
      dict = pickle.load(f, encoding='latin1')
    x = dict['data']
    y = dict['labels']
    x = x.astype(float)
    y = np.array(y)
  return x, y

def load_data():
  '''load all CIFAR-10 data and merge training batches'''
  R_ = np.full((10000, 1024), 125.3, dtype=np.float)
  R_dev = np.full((10000, 1024), 63.0, dtype=np.float)
  G_ = np.full((10000, 1024), 123.0, dtype=np.float)
  G_dev = np.full((10000, 1024), 62.1, dtype=np.float)
  B_ = np.full((10000, 1024), 113.9, dtype=np.float)
  B_dev = np.full((10000, 1024), 66.7, dtype=np.float)
  xs = []
  ys = []
  for i in range(1, 6):
    filename = 'cifar-10-batches-py/data_batch_' + str(i)
    X, Y = load_CIFAR10_batch(filename)
    if i == 1 :
        pp.pprint (X)
    X[:,:1024] -= R_
    X[:,:1024] /= R_dev
    X[:,1024:2048] -= G_
    X[:,1024:2048] /= G_dev
    X[:,2048:] -= B_
    X[:,2048:] /= B_dev
    if i == 1 :
        pp.pprint (X)
    xs.append(X)
    ys.append(Y)

  x_train = np.concatenate(xs)
  y_train = np.concatenate(ys)
  # transfer to one-hot for 10 class
  y_train = convertToOneHot(y_train)
  del xs, ys

  x_test, y_test = load_CIFAR10_batch('cifar-10-batches-py/test_batch')
  x_test[:,:1024] -= R_
  x_test[:,:1024] /= R_dev
  x_test[:,1024:2048] -= G_
  x_test[:,1024:2048] /= G_dev
  x_test[:,2048:] -= B_
  x_test[:,2048:] /= B_dev
  y_test = convertToOneHot(y_test)
  classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck']


  data_dict = {
    'images_train': x_train,
    'labels_train': y_train,
    'images_test': x_test,
    'labels_test': y_test,
    'classes': classes
  }
  return data_dict


def gen_batch(data, batch_size, num_iter):
  data = np.array(data)
  index = len(data)
  for i in range(num_iter):
    index += batch_size
    if (index + batch_size > len(data)):
      index = 0
      shuffled_indices = np.random.permutation(np.arange(len(data)))
      data = data[shuffled_indices]
    yield data[index:index + batch_size]
