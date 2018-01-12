import tensorflow as tf

from tsa.modeling.models.keypoint_params import *
from tsa.modeling.models.anchor import *
from tsa.modeling.models.keypoint_crop import *
import tsa.utils.data_path as dp
from tsa.utils.non_max_suppression import non_max_suppression
from tsa.utils.masks import *
from tsa.utils.merge_bbox_xml import parse_bbox_xml
from datetime import datetime
import time
import numpy as np
import os
import math
from termcolor import cprint, colored
from PIL import Image
from datascience import *
from tensorflow.python.ops import data_flow_ops

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python import debug as tf_debug

EPSILON = 1e-8
BOX_COUNT = 1
IMAGE_LENGTH = 200

REGION_ZONES = {
  'arm': {
    1: 0,
    2: 1,
    3: 2,
    4: 3
  },
  'thigh': {
    8: 0,
    9: 1,
    10: 2,
    11: 3,
    12: 4
  }
}


class ImagePreprocessor(object):
  def __init__(self, data_set, region, batch_size, mean_subtract):
    self.data_set = data_set
    self.region = region
    self.batch_size = batch_size
    self.mean_subtract = mean_subtract

  @property
  def image_shape(self):
    return (IMAGE_LENGTH, IMAGE_LENGTH)

  def distort_image(self, image, bbox):
    if 'train' not in self.data_set:
      return image, bbox
    original_bbox = bbox

    if self.region == 'arm':
      a_0 = tf.random_uniform([1], 0.85, 1.0)[0]
      a_2 = 0
      b_1 = tf.random_uniform([1], 0.85, 1.0)[0]
      b_2 = tf.random_uniform([1], -33, 33)[0]
    else:
      a_0 = tf.random_uniform([1], .80, 1.20)[0]
      a_2 = tf.random_uniform([1], -15, 15)[0]
      b_1 = tf.random_uniform([1], .80, 1.20)[0]
      b_2 = tf.random_uniform([1], -15, 15)[0]

    transformation = [a_0, 0, a_2, 0, b_1, b_2, 0, 0]
    image = tf.contrib.image.transform(
      image, transformation, interpolation='NEAREST')
    bbox = tf.cast(bbox, tf.float32)

    x_min, y_min, x_max, y_max = tf.split(
      value=tf.reshape(bbox, (-1, 4)), num_or_size_splits=4, axis=1)
    x_min = tf.minimum(tf.maximum(x_min / a_0 - a_2, 0), self.image_shape[1])
    y_min = tf.minimum(tf.maximum(y_min / b_1 - b_2, 0), self.image_shape[0])
    x_max = tf.minimum(tf.maximum(x_max / a_0 - a_2, 0), self.image_shape[1])
    y_max = tf.minimum(tf.maximum(y_max / b_1 - b_2, 0), self.image_shape[0])
    bbox = tf.reshape(tf.stack((x_min, y_min, x_max, y_max), axis=1), (-1, ))
    bbox = tf.cast(bbox, tf.int64)
    bbox = tf.cast(tf.not_equal(original_bbox, 0), tf.int64) * bbox
    return image, bbox

  def minibatch(self):
    with tf.name_scope('batch_processing'):
      images = []
      bboxes = []
      labels = []
      slcs = []

      record_input = data_flow_ops.RecordInput(
        file_pattern=dp.tf_records(self.data_set,
                                   '{}_localization'.format(self.region)),
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
        image, bbox, dim, label, slc = self._parse_example_proto(value, i)
        image = tf.image.resize_images(image, self.image_shape)
        bbox = tf.cast(
          tf.cast(bbox, tf.float32) * float(IMAGE_LENGTH) / tf.cast(
            dim, tf.float32), tf.int64)
        image, bbox = self.distort_image(image, bbox)
        unmaked_image = image
        bbox = tf.cast(bbox, tf.int32)
        bbox_mask = tf.ones((bbox[3] - bbox[1], bbox[2] - bbox[0]))
        bbox_mask_left = tf.zeros((bbox[3] - bbox[1], bbox[0]))
        bbox_mask_right = tf.zeros((bbox[3] - bbox[1], IMAGE_LENGTH - bbox[2]))
        bbox_mask_top = tf.zeros((bbox[1], IMAGE_LENGTH))
        bbox_mask_bottom = tf.zeros((IMAGE_LENGTH - bbox[3], IMAGE_LENGTH))
        bbox_mask = tf.concat(
          (bbox_mask_left, bbox_mask, bbox_mask_right), axis=1)
        bbox_mask = tf.concat(
          (bbox_mask_top, bbox_mask, bbox_mask_bottom), axis=0)
        bbox_mask *= 150
        image = image[:, :, 0]
        image -= self.mean_subtract
        image += bbox_mask
        image = tf.minimum(image, 255)
        bbox_mask.set_shape((IMAGE_LENGTH, IMAGE_LENGTH))
        images.append(image)
        bbox = tf.cast(bbox, tf.int64)
        bboxes.append(bbox)
        labels.append(label)
        slcs.append(slc)
      images = tf.parallel_stack(images)

      images = tf.reshape(
        images,
        shape=[self.batch_size, self.image_shape[0], self.image_shape[1], -1])
      bboxes = tf.reshape(bboxes, (self.batch_size, BOX_COUNT, 4))
      x_min, y_min, x_max, y_max = tf.split(
        value=tf.reshape(bboxes, (-1, 4)), num_or_size_splits=4, axis=1)
      normalized_boxes = tf.cast(
        tf.reshape(
          tf.stack((y_min, x_min, y_max, x_max), axis=1),
          (self.batch_size, BOX_COUNT, 4)), tf.float32) / float(IMAGE_LENGTH)
      # images = tf.image.draw_bounding_boxes(images, normalized_boxes)
      labels = tf.reshape(labels, (self.batch_size, 1))
      slcs = tf.reshape(slcs, (self.batch_size, ))
      return images, labels, slcs

  def _parse_example_proto(self, example_serialized, i):
    with tf.name_scope('parse_example'):
      features = {
        'image': tf.FixedLenFeature([], dtype=tf.string),
        'bbox': tf.FixedLenFeature([4 * BOX_COUNT], dtype=tf.int64),
        'dim': tf.FixedLenFeature([1], dtype=tf.int64),
        'label': tf.FixedLenFeature([1], dtype=tf.int64),
        'slc': tf.FixedLenFeature([1], dtype=tf.int64),
      }
      features = tf.parse_single_example(example_serialized, features=features)
      dim = features['dim']
      image = tf.reshape(
        tf.decode_raw(features['image'], tf.float32),
        tf.cast(
          tf.reshape(
            tf.stack((dim, dim, tf.constant([1], dtype=tf.int64))), (-1, )),
          tf.int32))
      return image, features['bbox'], dim, features['label'], features['slc']


class LocalizationModel(object):
  def __init__(self, region, batch_size):
    self.region = region
    self.batch_size = batch_size
    self.parameters = dict()
    self.weights_loaded = False

  def inference(self, images, slcs, validate=False, test=False):
    reuse = validate

    vgg_feature_maps = self._vgg16_feature_maps(reuse)(images[:, :, :, 0])
    flattened_feature_maps = tf.layers.flatten(vgg_feature_maps)
    fc_1 = self._fully_connected_layer(
      'fc_1', flattened_feature_maps.get_shape()[1].value, 1000, reuse,
      True)(flattened_feature_maps)
    fc_1 = tf.concat((fc_1, tf.one_hot(slcs, depth=16)), axis=1)
    fc_2 = self._fully_connected_layer('fc_2', 1000 + 16,
                                       len(REGION_ZONES[self.region]), reuse,
                                       True, tf.identity)(fc_1)
    return tf.reshape(fc_2, (self.batch_size, len(REGION_ZONES[self.region])))

  def load_weights(self, sess):
    assert not self.weights_loaded
    weights = np.load(dp.pretrained_weights('vgg16'))
    for layer in [
        'conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2',
        'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2',
        'conv5_3'
    ]:
      for var_type in [('weights', 'W'), ('biases', 'b')]:
        param_key = 'vgg16/{0}/{1}:0'.format(layer, var_type[0])
        weight_key = layer + '_' + var_type[1]
        sess.run(self.parameters[param_key].assign(weights[weight_key]))
    self.weights_loaded = True

  def _vgg16_feature_maps(self, reuse):
    def _op(input_):
      with tf.variable_scope('vgg16', reuse=reuse) as scope:
        reshaped = tf.stack([input_, input_, input_], axis=3)

        conv1_1 = self._conv_layer('conv1_1', [3, 3, 3, 64], reuse,
                                   False)(reshaped)
        conv1_2 = self._conv_layer('conv1_2', [3, 3, 64, 64], reuse,
                                   False)(conv1_1)
        pool1 = self._pool_layer('pool1')(conv1_2)
        conv2_1 = self._conv_layer('conv2_1', [3, 3, 64, 128], reuse,
                                   False)(pool1)
        conv2_2 = self._conv_layer('conv2_2', [3, 3, 128, 128], reuse,
                                   False)(conv2_1)
        pool2 = self._pool_layer('pool2')(conv2_2)
        conv3_1 = self._conv_layer('conv3_1', [3, 3, 128, 256], reuse,
                                   False)(pool2)
        conv3_2 = self._conv_layer('conv3_2', [3, 3, 256, 256], reuse,
                                   False)(conv3_1)
        conv3_3 = self._conv_layer('conv3_3', [3, 3, 256, 256], reuse,
                                   False)(conv3_2)
        pool3 = self._pool_layer('pool3')(conv3_3)
        conv4_1 = self._conv_layer('conv4_1', [3, 3, 256, 512], reuse,
                                   False)(pool3)
        conv4_2 = self._conv_layer('conv4_2', [3, 3, 512, 512], reuse,
                                   False)(conv4_1)
        conv4_3 = self._conv_layer('conv4_3', [3, 3, 512, 512], reuse,
                                   False)(conv4_2)
        pool4 = self._pool_layer('pool4')(conv4_3)
        conv5_1 = self._conv_layer('conv5_1', [3, 3, 512, 512], reuse,
                                   True)(pool4)
        conv5_2 = self._conv_layer('conv5_2', [3, 3, 512, 512], reuse,
                                   True)(conv5_1)
        conv5_3 = self._conv_layer('conv5_3', [3, 3, 512, 512], reuse,
                                   True)(conv5_2)
        return conv5_3

    return _op

  def _conv_layer(self,
                  layer_name,
                  kernel_shape,
                  reuse,
                  trainable,
                  initial_bias=0,
                  activation=tf.nn.relu):
    def _op(input_):
      with tf.variable_scope(layer_name, reuse=reuse) as scope:
        kernel = tf.get_variable(
          'weights',
          shape=kernel_shape,
          initializer=tf.truncated_normal_initializer(
            dtype=tf.float32, stddev=1e-2),
          dtype=tf.float32,
          trainable=trainable)
        conv = tf.nn.conv2d(input_, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable(
          'biases',
          shape=[kernel_shape[-1]],
          initializer=tf.constant_initializer(initial_bias, dtype=tf.float32),
          dtype=tf.float32,
          trainable=trainable)
        out = tf.nn.bias_add(conv, biases)
        conv_output = activation(out, name=scope.name)
        self._add_param(kernel)
        self._add_param(biases)
        return conv_output

    return _op

  def _fully_connected_layer(self,
                             layer_name,
                             in_dim,
                             out_dim,
                             reuse,
                             trainable,
                             activation=tf.nn.relu,
                             keep_prob=1,
                             initial_bias=0):
    def _op(input_):
      with tf.variable_scope(layer_name, reuse=reuse) as scope:
        weights = tf.get_variable(
          'weights',
          shape=[in_dim, out_dim],
          initializer=tf.truncated_normal_initializer(
            dtype=tf.float32, stddev=1e-2),
          dtype=tf.float32,
          trainable=trainable)
        biases = tf.get_variable(
          'biases',
          shape=[out_dim],
          initializer=tf.constant_initializer(initial_bias, dtype=tf.float32),
          dtype=tf.float32,
          trainable=trainable)
        out = activation(tf.matmul(input_, weights) + biases, name=scope.name)
        return out

    return _op

  def _pool_layer(self, layer_name):
    def _op(input_):
      return tf.nn.max_pool(
        input_,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name=layer_name)

    return _op

  def _add_param(self, param):
    if param.name not in self.parameters:
      self.parameters[param.name] = param


def train_func(total_loss, global_step):
  opt = tf.train.AdamOptimizer(learning_rate=1e-6)
  grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  with tf.control_dependencies([apply_gradient_op]):
    train_op = tf.no_op(name='train')

  return train_op


def loss_func(logits, labels, region):
  logits = tf.cast(logits, tf.float32)
  logits = tf.reshape(logits, (-1, len(REGION_ZONES[region])))
  labels = tf.reshape(labels, (-1, ))
  return tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits))


def train(region, batch_size, mean_subtract):
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    with tf.device('/cpu:0'):
      host_images, host_labels, host_slcs = ImagePreprocessor(
        'train_0', region, batch_size, mean_subtract).minibatch()

      images_shapes = host_images.get_shape()
      labels_shape = host_labels.get_shape()
      slcs_shape = host_slcs.get_shape()

      ### Validation Code ###
      validation_images, validation_labels, validation_slcs = ImagePreprocessor(
        'validate_0', region, batch_size, mean_subtract).minibatch()

    with tf.device('/gpu:0'):
      gpu_compute_stage = data_flow_ops.StagingArea(
        [tf.float32, tf.int64, tf.int64],
        shapes=[images_shapes, labels_shape, slcs_shape])
      # The CPU-to-GPU copy is triggered here.
      gpu_transfer_op = gpu_compute_stage.put(
        [host_images, host_labels, host_slcs])

      received_values = gpu_compute_stage.get()
      images, labels, slcs = received_values

      model = LocalizationModel(region, batch_size)
      logits = model.inference(images, slcs)

      loss = loss_func(logits, labels, region)

      ### Validation Code ###
      validation_logits = model.inference(
        validation_images, validation_slcs, validate=True)

      validation_loss = loss_func(validation_logits, validation_labels, region)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = train_func(loss, global_step)
    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=False))

    sess.run(init)
    model.load_weights(sess)
    if FLAGS.checkpoint_dir:
      saver.restore(
        sess,
        tf.train.get_checkpoint_state(
          dp.train_dir(FLAGS.checkpoint_dir)).model_checkpoint_path)

    sess.run(gpu_transfer_op)

    assert model.weights_loaded

    with open(dp.train_dir(FLAGS.train_dir + '/train_log.txt'), 'w') as f:
      duration = 0
      for step in xrange(FLAGS.max_steps):
        start_time = time.time()
        _, _, loss_value = sess.run([train_op, gpu_transfer_op, loss])
        duration += time.time() - start_time
        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
          num_examples_per_step = batch_size
          examples_per_sec = num_examples_per_step / (duration / 10.0)
          sec_per_batch = duration / 10.0

          format_str = ('%s: step %d, loss = %.8f '
                        '(%.1f examples/sec; %.3f sec/batch)')
          log_text = format_str % (datetime.now(), step, loss_value,
                                   examples_per_sec, sec_per_batch)
          f.write(log_text + '\n')
          print(log_text)
          duration = 0

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
          checkpoint_path = os.path.join(
            dp.train_dir(FLAGS.train_dir), 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

          total_loss = 0
          num_iter = 30
          for _ in range(num_iter):
            loss_of_predictions = sess.run(validation_loss)
            total_loss += loss_of_predictions

          log_text = '%s: ' % datetime.now() + 'validation loss = %.5f' % (
            total_loss / num_iter)
          f.write(log_text + '\n')
          f.flush()
          print(colored(log_text, 'yellow', attrs=['bold']))


def main(argv=None):  # pylint: disable=unused-argument
  train_dir = dp.train_dir(FLAGS.train_dir)
  if tf.gfile.Exists(train_dir):
    raise Exception('train_dir: {0} exists'.format(train_dir))
  tf.gfile.MakeDirs(train_dir)
  np.random.seed(FLAGS.random_seed)
  tf.set_random_seed(FLAGS.random_seed)
  train(FLAGS.region, FLAGS.batch_size, FLAGS.mean_subtract)


if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS

  tf.app.flags.DEFINE_string('train_dir', None,
                             """Directory where to write event logs """
                             """and checkpoint.""")
  tf.app.flags.DEFINE_string('region', None, 'arm or thigh')
  tf.app.flags.DEFINE_float('mean_subtract', None, 'arm or thigh')
  tf.app.flags.DEFINE_string('checkpoint_dir', None,
                             """Directory to initialize checkpoint from"""
                             """and checkpoint.""")
  tf.app.flags.DEFINE_integer('max_steps', 100000,
                              """Number of batches to run.""")
  tf.app.flags.DEFINE_integer('batch_size', 16, """Batch size""")
  tf.app.flags.DEFINE_integer('random_seed', 29,
                              """Random seed for np and tf""")

  tf.app.run()