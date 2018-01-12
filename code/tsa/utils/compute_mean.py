import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
import tsa.utils.data_path as dp
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_set', None, '')
tf.app.flags.DEFINE_string('file_format', None, '')
tf.app.flags.DEFINE_string('region', None, '')
tf.app.flags.DEFINE_integer('num_iters', 1000, '')


class ImagePreprocessor(object):
  def __init__(self, data_set, file_format, region, batch_size):
    self.data_set = data_set
    self.file_format = file_format
    self.region = region
    self.batch_size = batch_size

  @property
  def image_shape(self):
    return (600, 600)

  def distort_image(self, image):
    if 'train' not in self.data_set:
      return image

    a_0 = tf.random_uniform([1], .80, 1.20)[0]
    a_2 = tf.random_uniform([1], -50, 50)[0]
    b_1 = tf.random_uniform([1], .80, 1.20)[0]
    b_2 = tf.random_uniform([1], -50, 50)[0]

    transformation = [a_0, 0, a_2, 0, b_1, b_2, 0, 0]
    image = tf.contrib.image.transform(
      image, transformation, interpolation='NEAREST')
    return image

  def minibatch(self):
    with tf.name_scope('batch_processing'):
      images = []

      record_input = data_flow_ops.RecordInput(
        file_pattern=dp.tf_records(self.data_set,
                                   '{0}_crop_patch_full_{1}'.format(
                                     self.file_format, self.region)),
        seed=301,
        parallelism=64,
        buffer_size=5000,
        shift_ratio=0.2,
        batch_size=self.batch_size,
        name='record_input')
      records = record_input.get_yield_op()
      records = tf.split(records, self.batch_size, 0)
      records = [tf.reshape(record, []) for record in records]
      for i in xrange(self.batch_size):
        value = records[i]
        image, dim = self._parse_example_proto(value, i)
        image = tf.image.resize_images(image, self.image_shape)
        image = self.distort_image(image)
        image = image[:, :, 0]
        images.append(image)
      images = tf.parallel_stack(images)

      images = tf.reshape(
        images,
        shape=[self.batch_size, self.image_shape[0], self.image_shape[1], -1])
      return images

  def _parse_example_proto(self, example_serialized, i):
    with tf.name_scope('parse_example'):
      features = {
        'image': tf.FixedLenFeature([], dtype=tf.string),
        'dim': tf.FixedLenFeature([1], dtype=tf.int64),
      }
      features = tf.parse_single_example(example_serialized, features=features)
      dim = features['dim']
      image = tf.reshape(
        tf.decode_raw(features['image'], tf.float32),
        tf.cast(
          tf.reshape(
            tf.stack((dim, dim, tf.constant([1], dtype=tf.int64))), (-1, )),
          tf.int32))
      return image, dim


def _compute_mean(data_set, file_format, region, num_iters):
  ip = ImagePreprocessor(data_set, file_format, region, batch_size=16)
  images_tensor = ip.minibatch()
  with tf.Session() as sess:
    means = []
    for i in xrange(num_iters):
      images = sess.run(images_tensor)
      means.append(np.mean(images))
    print('{0} {1} {2} Mean: {3}'.format(data_set, file_format, region,
                                         np.mean(means)))


def main(argv=None):
  assert FLAGS.data_set and FLAGS.file_format and FLAGS.region and FLAGS.num_iters
  _compute_mean(FLAGS.data_set, FLAGS.file_format, FLAGS.region,
                FLAGS.num_iters)


if __name__ == '__main__':
  tf.app.run()
