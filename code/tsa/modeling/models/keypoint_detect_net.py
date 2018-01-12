import tensorflow as tf
from tensorflow.python.ops import data_flow_ops

from tsa.modeling.models.keypoint_params import *
from tsa.modeling.models.anchor import *
from tsa.modeling.models.focal_loss import loss_func
import tsa.utils.data_path as dp
from tsa.utils.non_max_suppression import non_max_suppression
from tsa.utils.masks import *
from tsa.utils.merge_bbox_xml import parse_bbox_xml, write_xml
from datetime import datetime
import time
import numpy as np
import os
import math
from termcolor import cprint, colored
from PIL import Image
from datascience import *

from six.moves import xrange  # pylint: disable=redefined-builtin

EPSILON = 1e-8
BOX_COUNT = 5


class ImagePreprocessor(object):
  def __init__(self, data_set, keypoint, batch_size):
    self.data_set = data_set
    self.keypoint = keypoint
    self.batch_size = batch_size

  @property
  def image_shape(self):
    return (660, 512)

  def distort_image(self, image, bbox):
    if 'train' not in self.data_set:
      return image, bbox

    a_0 = tf.random_uniform([1], .80, 1.20)[0]
    a_2 = tf.random_uniform([1], -100, 100)[0]
    b_1 = tf.random_uniform([1], .80, 1.20)[0]
    b_2 = tf.random_uniform([1], -60, 60)[0]

    transformation = [a_0, 0, a_2, 0, b_1, b_2, 0, 0]
    distored_image = tf.contrib.image.transform(
      image, transformation, interpolation='NEAREST')
    bbox = tf.cast(bbox, tf.float32)
    distored_bbox = [
      bbox[0] / a_0 - a_2, bbox[1] / b_1 - b_2, bbox[2] / a_0 - a_2,
      bbox[3] / b_1 - b_2
    ]

    image, bbox = distored_image, tf.cast(distored_bbox, tf.int64)

    random_mask = tf.distributions.Bernoulli(probs=[1 / 2.]).sample()
    mask_type = tf.distributions.Categorical(probs=5 * [1 / 5.]).sample()
    masked_images = []
    masked_images.append(image * vertical_stripe_mask((660, 512), (4, 4), (4,
                                                                           4)))
    masked_images.append(image * vertical_stripe_mask((660, 512), (4, 4),
                                                      (4, 15)))
    masked_images.append(image * horizontal_stripe_mask((660, 512), (4, 4),
                                                        (4, 4)))
    masked_images.append(image * horizontal_stripe_mask((660, 512), (4, 4),
                                                        (15, 4)))
    masked_images.append(image * squares_mask((660, 512), (10, 10), (10, 25)))

    image = tf.cond(
      tf.logical_and(tf.cast(random_mask, tf.bool)[0], tf.equal(mask_type, 0)),
      lambda: masked_images[0], lambda: image)
    image = tf.cond(
      tf.logical_and(tf.cast(random_mask, tf.bool)[0], tf.equal(mask_type, 1)),
      lambda: masked_images[1], lambda: image)
    image = tf.cond(
      tf.logical_and(tf.cast(random_mask, tf.bool)[0], tf.equal(mask_type, 2)),
      lambda: masked_images[2], lambda: image)
    image = tf.cond(
      tf.logical_and(tf.cast(random_mask, tf.bool)[0], tf.equal(mask_type, 3)),
      lambda: masked_images[3], lambda: image)
    image = tf.cond(
      tf.logical_and(tf.cast(random_mask, tf.bool)[0], tf.equal(mask_type, 4)),
      lambda: masked_images[4], lambda: image)
    return image, bbox

  def minibatch(self):
    with tf.name_scope('batch_processing'):
      images = []
      bboxes = []

      record_input = data_flow_ops.RecordInput(
        file_pattern=dp.tf_records(self.data_set,
                                   '{0}_keypoint'.format(self.keypoint.name)),
        seed=301,
        parallelism=64,
        buffer_size=2000,
        shift_ratio=0.2,
        batch_size=self.batch_size,
        name='record_input')
      records = record_input.get_yield_op()
      records = tf.split(records, self.batch_size, 0)
      records = [tf.reshape(record, []) for record in records]
      for i in xrange(self.batch_size):
        value = records[i]
        image, bbox = self._parse_example_proto(value, i)
        bbox = bbox[:4]
        image, bbox = self.distort_image(image, bbox)
        image -= tf.reduce_mean(image, axis=[0, 1])
        images.append(image)
        bboxes.append(bbox)
      images = tf.parallel_stack(images)

      images = tf.reshape(
        images,
        shape=[self.batch_size, self.image_shape[0], self.image_shape[1], -1])
      bboxes = tf.reshape(bboxes, (self.batch_size, 1, 4))

      return images, bboxes

  def _parse_example_proto(self, example_serialized, i):
    with tf.name_scope('parse_example'):
      features = {
        'image': tf.FixedLenFeature([], dtype=tf.string),
        'bbox': tf.FixedLenFeature([4 * BOX_COUNT], dtype=tf.int64),
      }
      features = tf.parse_single_example(example_serialized, features=features)
      image = tf.reshape(
        tf.decode_raw(features['image'], tf.float32), self.image_shape)
      return image, features['bbox']


def vgg16_feature_map_shape(image_dim):
  out_height, out_width = image_dim
  for i in range(4):
    out_height = math.ceil(float(out_height) / 2)
    out_width = math.ceil(float(out_width) / 2)
  return out_height, out_width


class KeypointDetectModel(object):
  def __init__(self,
               batch_size,
               image_dim,
               anchor_kwargs={
                 'base_size': 16,
                 'ratios': [0.5, 1, 2],
                 'scales': 2**np.arange(3, 6),
                 'allowed_border': 0
               }):
    self.batch_size = batch_size
    self.parameters = dict()
    self.weights_loaded = False
    num_template_anchors, anchors, inside_img_inds = make_anchors(
      image_dim, vgg16_feature_map_shape(image_dim), **anchor_kwargs)
    self.num_template_anchors = num_template_anchors
    self.anchors = anchors
    self.inside_img_inds = inside_img_inds

  def inference(self, images, validate=False, test=False):
    reuse = validate

    vgg_feature_maps = self._vgg16_feature_maps(reuse)(images[:, :, :, 0])

    logit_output = self._classification_subnet_layer(reuse)(vgg_feature_maps)
    bbox_output = self._box_regression_subnet_layer(reuse)(vgg_feature_maps)

    return logit_output, bbox_output

  def anchor_targets(self, gt_boxes):
    pos_iou_thresh = 0.7
    neg_iou_thresh = 0.4

    box_iou = iou(self.anchors, gt_boxes)  # (batch, num_anchors, num_gt_boxes)

    # for each gt_box, get the anchor with the highest IOU
    max_iou_per_gt_box = tf.reduce_max(box_iou, axis=1)
    max_iou_per_gt_box_indices = tf.argmax(box_iou, axis=1)

    # for each anchor, get the highest IOU it has with any gt_box
    max_iou_per_anchor = tf.reduce_max(box_iou, axis=2)
    max_iou_per_anchor_indices = tf.argmax(box_iou, axis=2)

    gt_cls_labels = -tf.ones(
      (self.batch_size, self.anchors.shape[0]), dtype=tf.int64)
    gt_cls_labels = tf.where(
      tf.greater(max_iou_per_anchor, pos_iou_thresh),
      tf.ones_like(gt_cls_labels), gt_cls_labels)
    gt_cls_labels = tf.where(
      tf.less(max_iou_per_anchor, neg_iou_thresh),
      tf.zeros_like(gt_cls_labels), gt_cls_labels)

    anchor_gt_box = tf.gather_nd(
      gt_boxes,
      tf.reshape(
        tf.transpose(
          tf.stack((tf.reshape(
            tf.ones_like(max_iou_per_anchor_indices) * tf.reshape(
              tf.range(max_iou_per_anchor_indices.shape[0], dtype=tf.int64),
              (-1, 1)), [-1]), tf.reshape(max_iou_per_anchor_indices, [-1])))),
        (self.batch_size, -1, 2)))

    # mask and remove anchors whose targets are [0, 0, 0, 0], which
    # is the placeholder ground truth box
    mask = tf.where(
      tf.greater(tf.reduce_sum(anchor_gt_box, axis=2), 0),
      tf.ones_like(gt_cls_labels), tf.zeros_like(gt_cls_labels))
    gt_cls_labels = mask * gt_cls_labels

    # treat as if constant data
    anchor_gt_box = tf.stop_gradient(anchor_gt_box)
    gt_cls_labels = tf.stop_gradient(gt_cls_labels)

    return anchor_gt_box, gt_cls_labels

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
                                   True)(pool3)
        conv4_2 = self._conv_layer('conv4_2', [3, 3, 512, 512], reuse,
                                   True)(conv4_1)
        conv4_3 = self._conv_layer('conv4_3', [3, 3, 512, 512], reuse,
                                   True)(conv4_2)
        pool4 = self._pool_layer('pool4')(conv4_3)
        conv5_1 = self._conv_layer('conv5_1', [3, 3, 512, 512], reuse,
                                   True)(pool4)
        conv5_2 = self._conv_layer('conv5_2', [3, 3, 512, 512], reuse,
                                   True)(conv5_1)
        conv5_3 = self._conv_layer('conv5_3', [3, 3, 512, 512], reuse,
                                   True)(conv5_2)
        return conv5_3

    return _op

  def _classification_subnet_layer(self, reuse):
    def _op(vgg_feature_maps):
      with tf.variable_scope('classification_subnet', reuse=reuse) as scope:
        conv1 = self._conv_layer('conv1', [3, 3, 512, 512], reuse,
                                 True)(vgg_feature_maps)
        initial_bias = -np.log((1 - 0.01) / 0.01)
        conv2 = self._conv_layer('conv2',
                                 [3, 3, 512, self.num_template_anchors], reuse,
                                 True, initial_bias, tf.identity)(conv1)
        reshaped_output = tf.reshape(conv2, (self.batch_size, -1))
        return tf.gather(reshaped_output, self.inside_img_inds, axis=1)

    return _op

  def _box_regression_subnet_layer(self, reuse):
    def _op(vgg_feature_maps):
      with tf.variable_scope('box_regression_subnet', reuse=reuse) as scope:
        conv1 = self._conv_layer('conv1', [3, 3, 512, 512], reuse,
                                 True)(vgg_feature_maps)
        conv2 = self._conv_layer(
          'conv2', [3, 3, 512, 4 * self.num_template_anchors],
          reuse,
          True,
          activation=tf.identity)(conv1)
        reshaped_output = tf.reshape(conv2, (self.batch_size, -1, 4))
        return tf.gather(reshaped_output, self.inside_img_inds, axis=1)

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
  opt = tf.train.AdamOptimizer(learning_rate=0.000001)  #0.00001)
  grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  with tf.control_dependencies([apply_gradient_op]):
    train_op = tf.no_op(name='train')

  return train_op


def train(keypoint, batch_size, random_seed):

  with tf.Graph().as_default():
    tf.set_random_seed(random_seed)
    global_step = tf.contrib.framework.get_or_create_global_step()

    with tf.device('/cpu:0'):
      host_images, host_bboxes = ImagePreprocessor('train_0', keypoint,
                                                   batch_size).minibatch()

      images_shapes = host_images.get_shape()
      bboxes_shape = host_bboxes.get_shape()

      ### Validation Code ###
      validation_images, validation_gt_boxes = ImagePreprocessor(
        'validate_0', keypoint, batch_size).minibatch()

    with tf.device('/gpu:0'):
      gpu_compute_stage = data_flow_ops.StagingArea(
        [tf.float32, tf.int64], shapes=[images_shapes, bboxes_shape])
      # The CPU-to-GPU copy is triggered here.
      gpu_transfer_op = gpu_compute_stage.put([host_images, host_bboxes])

      received_values = gpu_compute_stage.get()
      images, gt_boxes = received_values[0], received_values[1]

      model = KeypointDetectModel(
        batch_size=batch_size,
        image_dim=(660, 512),
        anchor_kwargs=keypoint.anchor_kwargs)
      logits, bboxes = model.inference(images)
      anchor_gt_box, gt_cls_labels = model.anchor_targets(gt_boxes)

      loss = loss_func(anchor_gt_box, bboxes, gt_cls_labels, logits,
                       model.anchors)

      ### Validation Code ###
      validation_logits, validation_bboxes = model.inference(
        validation_images, validate=True)
      validation_anchor_gt_box, validation_gt_cls_labels = model.anchor_targets(
        validation_gt_boxes)

      validation_loss = loss_func(validation_anchor_gt_box, validation_bboxes,
                                  validation_gt_cls_labels, validation_logits,
                                  model.anchors)

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

    sess.run(gpu_transfer_op)
    duration = 0
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, _, loss_value = sess.run([train_op, gpu_transfer_op, loss])
      duration += time.time() - start_time
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = batch_size
        examples_per_sec = num_examples_per_step / (duration / 10.0)
        sec_per_batch = duration / 10.

        format_str = ('%s: step %d, loss = %.8f '
                      '(%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (datetime.now(), step, loss_value, examples_per_sec,
                            sec_per_batch))
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

        print('%s: ' % datetime.now() + colored(
          'validation loss = %.5f' % (total_loss / num_iter),
          'yellow',
          attrs=['bold']))


def main(argv=None):  # pylint: disable=unused-argument
  train_dir = dp.train_dir(FLAGS.train_dir)
  if tf.gfile.Exists(train_dir):
    raise Exception('train_dir: {0} exists'.format(train_dir))
  tf.gfile.MakeDirs(train_dir)
  np.random.seed(FLAGS.random_seed)
  tf.set_random_seed(FLAGS.random_seed)

  if FLAGS.keypoint == 'face':
    keypoint = FACE
  if FLAGS.keypoint == 'butt':
    keypoint = BUTT

  train(keypoint, FLAGS.batch_size, FLAGS.random_seed)


if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS

  tf.app.flags.DEFINE_string('train_dir',
                             dp.train_dir(),
                             """Directory where to write event logs """
                             """and checkpoint.""")
  tf.app.flags.DEFINE_string('keypoint', None, 'face or butt')
  tf.app.flags.DEFINE_integer('max_steps', 20000,
                              """Number of batches to run.""")
  tf.app.flags.DEFINE_integer('batch_size', 10, """Batch size""")
  tf.app.flags.DEFINE_integer('random_seed', 29,
                              """Random seed for np and tf""")
  tf.app.run()
