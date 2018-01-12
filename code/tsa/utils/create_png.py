import tensorflow as tf
from datascience import *
import tsa.utils.data_path as dp
from tsa.utils.read_data import read_header
import tsa.utils.constants as CONST
import numpy as np
import scipy.misc
import os
import shutil

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_set', None, """data_set""")
tf.app.flags.DEFINE_string('file_format', None, """file_format""")


def _create_pngs(data_set, file_format):
  if file_format == 'aps':
    num_slcs, indices = 16, range(16)
  elif file_format == 'a3daps':
    num_slcs, indices = 64, [0]
  else:
    raise Exception('Invalid file format. Must be either aps or a3daps')
  ids = Table().read_table(dp.ids_by_data_set(data_set)).column('Id')
  for _id in ids:
    in_file = '{0}/data/raw/{1}/{2}.{1}'.format(dp.LARGE_DATA_BIN,
                                                       file_format, _id)
    header = read_header(in_file)
    with open(in_file, 'rb') as f:
      f.seek(CONST.RAW_HEADER_LENGTH)
      image = np.fromfile(
        f,
        dtype=np.uint16,
        count=num_slcs * CONST.A3DAPS_HEIGHT * CONST.A3DAPS_WIDTH)
    image = image.reshape(num_slcs, CONST.A3DAPS_HEIGHT,
                          CONST.A3DAPS_WIDTH).copy()
    for index in indices:
      image_slice = image[index, :, :]
      image_slice = image_slice.astype(np.float32)
      image_slice *= header['data_scale_factor']
      image_slice = np.flip(image_slice, axis=0)
      scipy.misc.imsave('{0}/data/raw/{1}_png/{2}/{3}.png'.format(
        dp.LARGE_DATA_BIN, file_format, index, str(index) + '_' + _id),
                        image_slice)


def main(argv=None):
  assert FLAGS.data_set and FLAGS.file_format, 'Must specify data_set and file_format'
  _create_pngs(FLAGS.data_set, FLAGS.file_format)


if __name__ == '__main__':
  tf.app.run()