from tsa.utils import data_path as dp
from datascience import *
import numpy as np
import json

TRAIN_PERCENT = 0.75


def partition_stage1_data():
  np.random.seed(42)
  stage1_labels = Table().read_table(dp.stage1_labels())
  unique_ids = list({i.split('_')[0] for i in stage1_labels.column('Id')})
  np.random.shuffle(unique_ids)
  split_index = int(len(unique_ids) * TRAIN_PERCENT)
  train_ids = unique_ids[:split_index]
  test_ids = unique_ids[split_index:]

  Table().with_column('Id', train_ids).to_csv(dp.ids_by_data_set('train'))
  Table().with_column('Id', test_ids).to_csv(dp.ids_by_data_set('validate'))

  train_ids = set(train_ids)

  train_labels, test_labels = {}, {}
  for i in range(stage1_labels.num_rows):
    full_id = stage1_labels.row(i).item(0)
    label = int(stage1_labels.row(i).item(1))
    split_id = full_id.split('_')
    id, zone = split_id[0], split_id[1].replace('Zone', '')

    if id in train_ids:
      if id not in train_labels:
        train_labels[id] = {zone: label}
      else:
        train_labels[id][zone] = label
    else:
      if id not in test_labels:
        test_labels[id] = {zone: label}
      else:
        test_labels[id][zone] = label

  with open(dp.labels_by_data_set('train', 'json'), 'w') as f:
    json.dump(train_labels, f, indent=4)
  with open(dp.labels_by_data_set('validate', 'json'), 'w') as f:
    json.dump(test_labels, f, indent=4)

def further_partition_data():
  train_ids = Table().read_table(dp.ids_by_data_set('train')).column('Id').tolist()
  validate_ids = Table().read_table(dp.ids_by_data_set('validate')).column('Id').tolist()

  split_index = len(train_ids) // 3
  
  validate_0_ids = validate_ids
  validate_1_ids = train_ids[:split_index]
  validate_2_ids = train_ids[split_index:2*split_index]
  validate_3_ids = train_ids[2*split_index:]

  train_0_ids = train_ids
  train_1_ids = validate_0_ids + validate_2_ids + validate_3_ids
  train_2_ids = validate_0_ids + validate_1_ids + validate_3_ids
  train_3_ids = validate_0_ids + validate_1_ids + validate_2_ids

  for i in range(4):
    train_name, validate_name = 'train_{}'.format(i), 'validate_{}'.format(i)
    Table().with_column('Id', eval(train_name + '_ids')).to_csv(dp.ids_by_data_set(train_name))
    Table().with_column('Id', eval(validate_name + '_ids')).to_csv(dp.ids_by_data_set(validate_name))

if __name__ == '__main__':
  # partition_stage1_data()
  further_partition_data()