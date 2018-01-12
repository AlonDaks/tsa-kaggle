import os
from datascience import *
import datetime
from tsa.config import REPO_HOME_PATH, LARGE_DATA_BIN


def stage1_labels(format='csv'):
  return '{0}/data/stage1_labels.csv'.format(REPO_HOME_PATH)


def labels_by_data_set(data_set, format='csv'):
  _validate_data_set(data_set, only_with_labels=True)
  if format not in ['csv', 'json']:
    raise ValueError("format must be either 'csv' or 'json'")
  return '{0}/data/{1}_labels.{2}'.format(REPO_HOME_PATH, data_set, format)


def ids_by_data_set(data_set):
  return '{0}/data/{1}_ids.csv'.format(REPO_HOME_PATH, data_set)


def tf_records(data_set, file_format):
  return '{0}/data/tf_records/{1}_{2}.tfrecords'.format(
    LARGE_DATA_BIN, data_set, file_format)


def train_dir(dir_name=None):
  if not dir_name:
    return '{0}/train_dir/train_{1}'.format(
      LARGE_DATA_BIN, datetime.datetime.now().strftime('%m-%d_%H:%M:%S'))
  return '{0}/train_dir/{1}'.format(LARGE_DATA_BIN, dir_name)


def pretrained_weights(model_name):
  return '{0}/data/pretrained_weights/{1}.npz'.format(REPO_HOME_PATH,
                                                      model_name)