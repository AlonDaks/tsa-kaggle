from datascience import *
import tsa.utils.data_path as dp


def test_partition_stage1_data():
  train_ids = set(Table().read_table(dp.ids_by_data_set('train')).column('Id'))
  validate_ids = set(Table().read_table(dp.ids_by_data_set('validate')).column('Id'))
  test_ids = set(Table().read_table(dp.ids_by_data_set('test')).column('Id'))

  assert train_ids.intersection(validate_ids) == set()
  assert train_ids.intersection(test_ids) == set()
  assert validate_ids.intersection(test_ids) == set()