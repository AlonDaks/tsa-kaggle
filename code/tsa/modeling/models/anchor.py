"""
This file incorporates material from https://github.com/rbgirshick/py-faster-rcnn (MIT License)
Copied below is the original license.

Faster R-CNN

The MIT License (MIT)

Copyright (c) 2015 Microsoft Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import tensorflow as tf
import numpy as np


def make_anchors(image_dim, feature_map_shape, base_size, ratios, scales,
                 allowed_border):
  template_anchors = generate_template_anchors(base_size, ratios, scales)
  all_anchors = generate_total_anchors(template_anchors, feature_map_shape,
                                       image_dim)

  inds_inside = (np.where((all_anchors[:, 0] >= -allowed_border) & (
    all_anchors[:, 1] >=
    -allowed_border) & (all_anchors[:, 2] < image_dim[1] + allowed_border) & (
      all_anchors[:, 3] < image_dim[0] + allowed_border))[0])

  anchors = all_anchors[inds_inside, :]
  return len(template_anchors), anchors, inds_inside


def iou(anchors, gt_boxes, scope=None):
  with tf.name_scope(scope, 'iou'):
    anchors = tf.cast(anchors, tf.float32)
    gt_boxes = tf.cast(gt_boxes, tf.float32)

    intersections = intersection(anchors, gt_boxes)  # (batch, #anchors, #gt)
    areas1 = area(anchors)  # (#anchors)
    areas1 = tf.tile(tf.expand_dims(areas1, 0), [tf.shape(gt_boxes)[0],
                                                 1])  # (batch, #anchors)
    areas2 = area(gt_boxes)  # (batch, #gt)
    unions = (
      tf.expand_dims(areas1, 2) + tf.expand_dims(areas2, 1) - intersections)
    return tf.where(
      tf.equal(intersections, 0.0),
      tf.zeros_like(intersections), tf.truediv(intersections, unions))


def intersection(anchors, gt_boxes, scope=None):
  with tf.name_scope(scope, 'intersection'):
    x_min1, y_min1, x_max1, y_max1 = tf.split(
      value=anchors, num_or_size_splits=4, axis=1)
    x_min1 = tf.tile(tf.expand_dims(x_min1, 0), [tf.shape(gt_boxes)[0], 1, 1])
    y_min1 = tf.tile(tf.expand_dims(y_min1, 0), [tf.shape(gt_boxes)[0], 1, 1])
    x_max1 = tf.tile(tf.expand_dims(x_max1, 0), [tf.shape(gt_boxes)[0], 1, 1])
    y_max1 = tf.tile(tf.expand_dims(y_max1, 0), [tf.shape(gt_boxes)[0], 1, 1])
    x_min2, y_min2, x_max2, y_max2 = tf.split(
      value=gt_boxes, num_or_size_splits=4, axis=2)

    all_pairs_min_ymax = tf.minimum(y_max1, tf.transpose(y_max2, [0, 2, 1]))
    all_pairs_max_ymin = tf.maximum(y_min1, tf.transpose(y_min2, [0, 2, 1]))
    intersect_heights = tf.maximum(0.0,
                                   all_pairs_min_ymax - all_pairs_max_ymin)
    all_pairs_min_xmax = tf.minimum(x_max1, tf.transpose(x_max2, [0, 2, 1]))
    all_pairs_max_xmin = tf.maximum(x_min1, tf.transpose(x_min2, [0, 2, 1]))
    intersect_widths = tf.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
    return intersect_heights * intersect_widths


def area(boxes, scope=None):
  with tf.name_scope(scope, 'area'):
    x_min, y_min, x_max, y_max = tf.split(
      value=boxes, num_or_size_splits=4, axis=tf.rank(boxes) - 1)
    return tf.squeeze((y_max - y_min) * (x_max - x_min),
                      [len(boxes.shape) - 1])


def _box2whctrs(boxes):
  boxes = tf.cast(boxes, tf.float32)

  x_min, y_min, x_max, y_max = tf.split(
    value=boxes, num_or_size_splits=4, axis=2)
  w = x_max - x_min + 1
  h = y_max - y_min + 1
  x_ctr = x_min + 0.5 * (w - 1)
  y_ctr = y_min + 0.5 * (h - 1)
  return tf.squeeze(tf.stack([w, h, x_ctr, y_ctr], axis=2), [-1])


def box_to_target(boxes, anchors, scope=None):
  with tf.name_scope(scope, 'box_to_target'):
    boxes = tf.cast(boxes, tf.float32)
    anchors = tf.cast(anchors, tf.float32)
    anchors = tf.tile(tf.expand_dims(anchors, 0), [tf.shape(boxes)[0], 1, 1])

    w, h, x_ctr, y_ctr = tf.split(
      value=_box2whctrs(boxes), num_or_size_splits=4, axis=2)
    w_a, h_a, x_ctr_a, y_ctr_a = tf.split(
      value=_box2whctrs(anchors), num_or_size_splits=4, axis=2)

    t_x_ctr = (x_ctr - x_ctr_a) / w_a
    t_y_ctr = (y_ctr - y_ctr_a) / h_a
    t_w = tf.log(w / w_a)
    t_h = tf.log(h / h_a)

    target = tf.squeeze(tf.stack([t_w, t_h, t_x_ctr, t_y_ctr], axis=2), [-1])

  return target


def target_to_box(targets, anchors, scope=None):
  with tf.name_scope(scope, 'target_to_box'):
    anchors = tf.cast(anchors, tf.float32)
    anchors = tf.tile(tf.expand_dims(anchors, 0), [tf.shape(targets)[0], 1, 1])

    t_w, t_h, t_x_ctr, t_y_ctr = tf.split(
      value=targets, num_or_size_splits=4, axis=2)
    w_a, h_a, x_ctr_a, y_ctr_a = tf.split(
      value=_box2whctrs(anchors), num_or_size_splits=4, axis=2)

    x_ctr = t_x_ctr * w_a + x_ctr_a
    y_ctr = t_y_ctr * h_a + y_ctr_a
    w = tf.exp(t_w) * w_a
    h = tf.exp(t_h) * h_a

    x_min = x_ctr - w / 2.0
    y_min = y_ctr - h / 2.0
    x_max = x_ctr + w / 2.0
    y_max = y_ctr + h / 2.0

    bboxes = tf.squeeze(tf.stack([x_min, y_min, x_max, y_max], axis=2), [-1])

  return bboxes


def generate_total_anchors(template_anchors,
                           feature_map_shape,
                           input_image_shape,
                           anchor_stride=None):
  if anchor_stride is None:
    anchor_stride = np.array(input_image_shape) // np.array(feature_map_shape)
  shift_x = np.arange(0, feature_map_shape[1]) * anchor_stride[1]
  shift_y = np.arange(0, feature_map_shape[0]) * anchor_stride[0]
  shift_x, shift_y = np.meshgrid(shift_x, shift_y)
  shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(),
                      shift_y.ravel())).transpose()
  A = template_anchors.shape[0]
  K = shifts.shape[0]
  all_anchors = (template_anchors.reshape((1, A, 4)) + shifts.reshape(
    (1, K, 4)).transpose((1, 0, 2)))
  all_anchors = all_anchors.reshape((K * A, 4))
  return all_anchors


def generate_template_anchors(base_size, ratios, scales):
  base_anchor = np.array([1, 1, base_size, base_size]) - 1
  ratio_anchors = _ratio_enum(base_anchor, ratios)
  anchors = np.vstack([
    _scale_enum(ratio_anchors[i, :], scales)
    for i in range(ratio_anchors.shape[0])
  ])
  return anchors


def _whctrs(anchor):
  w = anchor[2] - anchor[0] + 1
  h = anchor[3] - anchor[1] + 1
  x_ctr = anchor[0] + 0.5 * (w - 1)
  y_ctr = anchor[1] + 0.5 * (h - 1)
  return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
  ws = ws[:, np.newaxis]
  hs = hs[:, np.newaxis]
  anchors = np.hstack((x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                       x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)))
  return anchors


def _ratio_enum(anchor, ratios):
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  size = w * h
  size_ratios = size / ratios
  ws = np.round(np.sqrt(size_ratios))
  hs = np.round(ws * ratios)
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors


def _scale_enum(anchor, scales):
  w, h, x_ctr, y_ctr = _whctrs(anchor)
  ws = w * scales
  hs = h * scales
  anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
  return anchors
