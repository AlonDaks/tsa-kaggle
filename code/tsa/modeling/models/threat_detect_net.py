import tensorflow as tf
from tensorflow.python.ops import data_flow_ops

from tsa.modeling.models.anchor import *
from tsa.modeling.models.focal_loss import loss_func
import tsa.utils.data_path as dp
import time
from datetime import datetime
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
  def __init__(self, data_set, mode, file_format, region, batch_size,
               mean_subtract):
    self.data_set = data_set
    self.mode = mode
    self.file_format = file_format
    self.region = region
    self.batch_size = batch_size
    self.mean_subtract = mean_subtract

  @property
  def image_shape(self):
    return (600, 600)

  def distort_image(self, image, bbox):
    if 'train' not in self.data_set:
      return image, bbox
    original_bbox = bbox

    if self.region == 'arm':
      a_0 = tf.random_uniform([1], 0.85, 1.0)[0]
      a_2 = 0
      b_1 = tf.random_uniform([1], 0.85, 1.0)[0]
      b_2 = tf.random_uniform([1], -100, 100)[0]
    else:
      a_0 = tf.random_uniform([1], .80, 1.20)[0]
      a_2 = tf.random_uniform([1], -50, 50)[0]
      b_1 = tf.random_uniform([1], .80, 1.20)[0]
      b_2 = tf.random_uniform([1], -50, 50)[0]

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

    random_flip = tf.distributions.Bernoulli(probs=[0.5]).sample()
    x_min, y_min, x_max, y_max = tf.split(
      value=tf.reshape(bbox, (-1, 4)), num_or_size_splits=4, axis=1)
    flipped_bbox = tf.reshape(
      tf.stack(
        (self.image_shape[1] - x_max, y_min, self.image_shape[1] - x_min,
         y_max),
        axis=1), (-1, ))
    flipped_bbox = tf.cast(tf.not_equal(original_bbox, 0),
                           tf.int64) * flipped_bbox
    image, bbox = tf.cond(
      tf.cast(random_flip, tf.bool)[0],
      lambda: (tf.image.flip_left_right(image), flipped_bbox),
      lambda: (image, bbox))
    return image, bbox

  def minibatch(self):
    with tf.name_scope('batch_processing'):
      images = []
      bboxes = []

      record_input = data_flow_ops.RecordInput(
        file_pattern=dp.tf_records(
          self.data_set, '{0}_crop_patch_{1}_{2}'.format(
            self.file_format, self.mode, self.region)),
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
        image, bbox, dim = self._parse_example_proto(value, i)
        image = tf.image.resize_images(image, self.image_shape)
        bbox = tf.cast(
          tf.cast(bbox, tf.float32) * 600. / tf.cast(dim, tf.float32),
          tf.int64)
        image, bbox = self.distort_image(image, bbox)
        image = image[:, :, 0]
        image -= self.mean_subtract
        images.append(image)
        bboxes.append(bbox)
      images = tf.parallel_stack(images)

      images = tf.reshape(
        images,
        shape=[self.batch_size, self.image_shape[0], self.image_shape[1], -1])
      bboxes = tf.reshape(bboxes, (self.batch_size, 5, 4))
      return images, bboxes

  def _parse_example_proto(self, example_serialized, i):
    with tf.name_scope('parse_example'):
      features = {
        'image': tf.FixedLenFeature([], dtype=tf.string),
        'bbox': tf.FixedLenFeature([4 * BOX_COUNT], dtype=tf.int64),
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
      return image, features['bbox'], dim


def vgg16_feature_map_shape(image_dim, level):
  out_height, out_width = image_dim
  for i in range(level):
    out_height = math.ceil(float(out_height) / 2)
    out_width = math.ceil(float(out_width) / 2)
  return out_height, out_width


class ThreatDetectModel(object):
  def __init__(self,
               batch_size,
               image_dim,
               base_sizes,
               anchor_kwargs={
                 'ratios': [0.5, 1, 2],
                 'scales': 2**np.array([0, 1 / 3., 2 / 3.]),
                 'allowed_border': 0
               }):
    self.batch_size = batch_size
    self.parameters = dict()
    self.weights_loaded = False
    num_template_anchors, l3_anchors, l3_inside_img_inds = make_anchors(
      image_dim,
      vgg16_feature_map_shape(image_dim, 3),
      base_size=base_sizes[0],
      **anchor_kwargs)
    _, l4_anchors, l4_inside_img_inds = make_anchors(
      image_dim,
      vgg16_feature_map_shape(image_dim, 4),
      base_size=base_sizes[1],
      **anchor_kwargs)
    _, l5_anchors, l5_inside_img_inds = make_anchors(
      image_dim,
      vgg16_feature_map_shape(image_dim, 5),
      base_size=256,
      **anchor_kwargs)
    self.num_template_anchors = num_template_anchors
    self.anchors = np.vstack((l3_anchors, l4_anchors, l5_anchors))
    self.inside_img_inds = {
      'l3': l3_inside_img_inds,
      'l4': l4_inside_img_inds,
      'l5': l5_inside_img_inds
    }
    self.image_dim = image_dim

  def inference(self, images, validate=False, test=False):
    reuse = validate

    vgg_feature_maps = self._vgg16_feature_maps(reuse)(images[:, :, :, 0])
    p3, p4, p5 = self._fpn_feature_maps(
      num_channels=256, reuse=reuse)(vgg_feature_maps)

    p3_logit_output = self._classification_subnet_layer(
      layer='l3', reuse=reuse)(p3)
    p3_bbox_output = self._box_regression_subnet_layer(
      layer='l3', reuse=reuse)(p3)

    p4_logit_output = self._classification_subnet_layer(
      layer='l4', reuse=True)(p4)
    p4_bbox_output = self._box_regression_subnet_layer(
      layer='l4', reuse=True)(p4)

    p5_logit_output = self._classification_subnet_layer(
      layer='l5', reuse=True)(p5)
    p5_bbox_output = self._box_regression_subnet_layer(
      layer='l5', reuse=True)(p5)

    logit_output = tf.concat(
      (p3_logit_output, p4_logit_output, p5_logit_output), axis=1)
    bbox_output = tf.concat(
      (p3_bbox_output, p4_bbox_output, p5_bbox_output), axis=1)

    return logit_output, bbox_output

  def anchor_targets(self, gt_boxes):
    pos_iou_thresh = 0.5
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
                                   True)(pool2)
        conv3_2 = self._conv_layer('conv3_2', [3, 3, 256, 256], reuse,
                                   True)(conv3_1)
        conv3_3 = self._conv_layer('conv3_3', [3, 3, 256, 256], reuse,
                                   True)(conv3_2)
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
        pool5 = self._pool_layer('pool5')(conv5_3)

        c1 = pool1
        c2 = pool2
        c3 = pool3
        c4 = pool4
        c5 = pool5

        return c3, c4, c5

    return _op

  def _fpn_feature_maps(self, num_channels, reuse):
    def _op(c_feature_maps):
      with tf.variable_scope('fpn_feature_maps', reuse=reuse) as scope:
        c3, c4, c5 = c_feature_maps
        p5 = self._conv_layer(
          'c5_conv', [1, 1, 512, num_channels],
          reuse,
          True,
          activation=tf.identity)(c5)
        t4 = self._conv_layer(
          'c4_conv', [1, 1, 512, num_channels],
          reuse,
          True,
          activation=tf.identity)(c4)
        upsampled_5 = tf.image.resize_images(
          p5,
          vgg16_feature_map_shape(self.image_dim,
                                  4), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        p4 = self._conv_layer(
          'p4_conv', [3, 3, num_channels, num_channels],
          reuse,
          True,
          activation=tf.identity)(t4 + upsampled_5)
        t3 = self._conv_layer(
          'c3_conv', [1, 1, 256, num_channels],
          reuse,
          True,
          activation=tf.identity)(c3)
        upsampled_4 = tf.image.resize_images(
          p4,
          vgg16_feature_map_shape(self.image_dim,
                                  3), tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        p3 = self._conv_layer(
          'p3_conv', [3, 3, num_channels, num_channels],
          reuse,
          True,
          activation=tf.identity)(t3 + upsampled_4)

        return p3, p4, p5

    return _op

  def _classification_subnet_layer(self, layer, reuse):
    def _op(vgg_feature_maps):
      with tf.variable_scope('classification_subnet', reuse=reuse) as scope:
        conv1 = self._conv_layer('conv1', [3, 3, 256, 256], reuse,
                                 True)(vgg_feature_maps)
        conv2 = self._conv_layer('conv2', [3, 3, 256, 256], reuse, True)(conv1)
        conv3 = self._conv_layer('conv3', [3, 3, 256, 256], reuse, True)(conv2)
        conv4 = self._conv_layer('conv4', [3, 3, 256, 256], reuse, True)(conv3)
        conv5 = self._conv_layer(
          'conv5', [3, 3, 256, self.num_template_anchors],
          reuse,
          True,
          initial_bias=-np.log((1 - 0.01) / 0.01),
          activation=tf.identity)(conv4)
        reshaped_output = tf.reshape(conv5, (self.batch_size, -1))
        return tf.gather(reshaped_output, self.inside_img_inds[layer], axis=1)

    return _op

  def _box_regression_subnet_layer(self, layer, reuse):
    def _op(vgg_feature_maps):
      with tf.variable_scope('box_regression_subnet', reuse=reuse) as scope:
        conv1 = self._conv_layer('conv1', [3, 3, 256, 256], reuse,
                                 True)(vgg_feature_maps)
        conv2 = self._conv_layer('conv2', [3, 3, 256, 256], reuse, True)(conv1)
        conv3 = self._conv_layer('conv3', [3, 3, 256, 256], reuse, True)(conv2)
        conv4 = self._conv_layer('conv4', [3, 3, 256, 256], reuse, True)(conv3)
        conv5 = self._conv_layer(
          'conv5', [3, 3, 256, 4 * self.num_template_anchors],
          reuse,
          True,
          activation=tf.identity)(conv4)
        reshaped_output = tf.reshape(conv5, (self.batch_size, -1, 4))
        return tf.gather(reshaped_output, self.inside_img_inds[layer], axis=1)

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
  opt = tf.train.AdamOptimizer(learning_rate=1e-6)
  grads = opt.compute_gradients(total_loss)

  # Apply gradients.
  apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

  with tf.control_dependencies([apply_gradient_op]):
    train_op = tf.no_op(name='train')

  return train_op


def train(fold, mode, file_format, region, batch_size, mean_subtract):
  with tf.Graph().as_default():
    tf.set_random_seed(FLAGS.random_seed)
    global_step = tf.contrib.framework.get_or_create_global_step()

    with tf.device('/cpu:0'):
      host_images, host_bboxes = ImagePreprocessor(
        'train_{}'.format(fold), mode, file_format, region, batch_size,
        mean_subtract).minibatch()

      images_shapes = host_images.get_shape()
      bboxes_shape = host_bboxes.get_shape()

      ### Validation Code ###
      validation_images, validation_gt_boxes = ImagePreprocessor(
        'validate_{}'.format(fold), mode, file_format, region, batch_size,
        mean_subtract).minibatch()

    with tf.device('/gpu:0'):
      gpu_compute_stage = data_flow_ops.StagingArea(
        [tf.float32, tf.int64], shapes=[images_shapes, bboxes_shape])
      # The CPU-to-GPU copy is triggered here.
      gpu_transfer_op = gpu_compute_stage.put([host_images, host_bboxes])

      received_values = gpu_compute_stage.get()
      images, gt_boxes = received_values[0], received_values[1]

      if region == 'torso':
        base_sizes = [128, 256]
      if region == 'calf':
        base_sizes = [128, 128]
      if region == 'thigh':
        base_sizes = [128, 256]
      if region == 'arm':
        base_sizes = [64, 128]

      model = ThreatDetectModel(
        batch_size=batch_size, image_dim=(600, 600), base_sizes=base_sizes)
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
  assert FLAGS.region and FLAGS.mean_subtract is not None

  train_dir = dp.train_dir(FLAGS.train_dir)
  if tf.gfile.Exists(train_dir):
    raise Exception('train_dir: {0} exists'.format(train_dir))
  tf.gfile.MakeDirs(train_dir)
  np.random.seed(FLAGS.random_seed)
  tf.set_random_seed(FLAGS.random_seed)
  train(FLAGS.fold, FLAGS.mode, FLAGS.file_format, FLAGS.region,
        FLAGS.batch_size, FLAGS.mean_subtract)


if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_string('train_dir', None,
                             """Directory where to write event logs """
                             """and checkpoint.""")
  tf.app.flags.DEFINE_string('checkpoint_dir', None,
                             """Directory to initialize checkpoint from"""
                             """and checkpoint.""")
  tf.app.flags.DEFINE_string('fold', 0, """Train / Validation Fold""")
  tf.app.flags.DEFINE_integer('max_steps', 100000,
                              """Number of batches to run.""")
  tf.app.flags.DEFINE_integer('batch_size', 16, """Batch size""")
  tf.app.flags.DEFINE_string('region', None,
                             """Region instance of model is ran on""")
  tf.app.flags.DEFINE_integer('random_seed', 29,
                              """Random seed for np and tf""")
  tf.app.flags.DEFINE_float('mean_subtract', None,
                            """Random seed for np and tf""")
  tf.app.flags.DEFINE_string('file_format', 'aps', """aps or a3daps""")
  tf.app.flags.DEFINE_string('mode', 'full', """full, mask or pure""")

  tf.app.run()