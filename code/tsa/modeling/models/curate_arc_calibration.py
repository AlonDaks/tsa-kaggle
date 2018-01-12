import tensorflow as tf
import numpy as np
from datascience import *
from tsa.utils.non_max_suppression import non_max_suppression
from tsa.utils.arc import *
from tsa.modeling.models.threat_detect_inference import threat_detect_inference
from tsa.utils.merge_bbox_xml import parse_bbox_xml
import tsa.utils.data_path as dp
import os
from skimage.transform import resize

EPSILON = 1e-8
BOX_COUNT = 5


def _iou(boxes_a, boxes_b):
  boxes_a = boxes_a.astype(np.float32)
  boxes_b = boxes_b.astype(np.float32)

  intersections = _intersection(boxes_a, boxes_b)
  areas_a = _area(boxes_a)
  areas_b = _area(boxes_b)
  unions = (areas_a[:, np.newaxis] + areas_b[np.newaxis, :] - intersections)
  return np.where(intersections == 0,
                  np.zeros_like(intersections), intersections / unions)


def _intersection(boxes_a, boxes_b):
  x_min_a, y_min_a, x_max_a, y_max_a = np.split(
    boxes_a, indices_or_sections=4, axis=1)
  x_min_b, y_min_b, x_max_b, y_max_b = np.split(
    boxes_b, indices_or_sections=4, axis=1)

  all_pairs_min_ymax = np.minimum(y_max_a, np.transpose(y_max_b))
  all_pairs_max_ymin = np.maximum(y_min_a, np.transpose(y_min_b))
  intersect_heights = np.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
  all_pairs_min_xmax = np.minimum(x_max_a, np.transpose(x_max_b))
  all_pairs_max_xmin = np.maximum(x_min_a, np.transpose(x_min_b))
  intersect_widths = np.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
  return intersect_heights * intersect_widths


def _area(boxes):
  x_min, y_min, x_max, y_max = np.split(boxes, indices_or_sections=4, axis=1)
  return np.reshape((y_max - y_min) * (x_max - x_min), (-1, ))


def images_and_crop_by_id(_id):
  images, cropped_bbox = [], None
  for i in range(16):
    image, cropped_bbox = crop_and_pad(
      _id, file_format='aps', region='torso', slc=i, keypoint_pad=20)
    images.append(image)
  images = [resize(i, (600, 600), mode='constant') - 16 for i in images]
  images = np.concatenate([i[np.newaxis, :, :] for i in images])
  images = images[:, :, :, np.newaxis]
  return images, cropped_bbox


def get_arcs_and_ground_truth(data_set, zones):
  ids = Table().read_table(dp.ids_by_data_set(data_set)).column('Id')

  images_tensor = tf.placeholder(tf.float32, (None, 600, 600, 1))
  model = AnchorBabyModel(batch_size=16, image_dim=(600, 600))

  logits_tensor, bboxes_tensor = model.inference(images_tensor, test=True)
  logits_tensor = tf.sigmoid(logits_tensor)
  bboxes_tensor = target_to_box(bboxes_tensor, model.anchors)
  checkpoint_dir = '/home/alon/sda1/tsa-kaggle/train_dir/train_11-25_20:14:05'  #patch crop torso w/ masks
  xml_base_path = '/home/alon/Documents/tsa-kaggle/data/bbox/aps_merged_threats/{0}_{1}.xml'
  xml_base_zone_path = '/home/alon/Documents/tsa-kaggle/data/bbox/aps_threats/zone_{2}/{0}_{1}.xml'
  saver = tf.train.Saver()
  with tf.Session() as sess:
    saver.restore(
      sess,
      tf.train.get_checkpoint_state(checkpoint_dir).model_checkpoint_path)
    arcs_matrix, arcs_labels, arc_ids, arc_zone_labels = [], [], [], []
    for _id in ids:
      print(_id)
      images, cropped_bbox = images_and_crop_by_id(_id)
      logits, bboxes = sess.run(
        [logits_tensor, bboxes_tensor], feed_dict={images_tensor: images})
      arcs = build_arcs(bboxes, logits)
      for arc in arcs:
        matched = False
        for zone in zones:
          num_matches = 0
          original_scale_boxes = arc.original_scale_boxes(cropped_bbox)
          for i in range(original_scale_boxes.shape[0]):
            gt_xml_path = xml_base_zone_path.format(i, _id, zone)
            if os.path.exists(gt_xml_path):
              gt_bboxes = np.array(parse_bbox_xml(gt_xml_path))
              if gt_bboxes.size > 0:
                if np.max(
                    _iou(gt_bboxes, original_scale_boxes[i]
                         [np.newaxis])) > 0.5:
                  num_matches += 1
          arc_match = num_matches / np.sum(
            np.sum(original_scale_boxes, axis=1) > 0) >= 0.5
          if arc_match:
            matched = True
            arcs_matrix.append(arc.to_list())
            arcs_labels.append(True)
            arc_ids.append(_id)
            arc_zone_labels.append(zone)
            break
        if not matched:
          arcs_matrix.append(arc.to_list())
          arcs_labels.append(False)
          arc_ids.append(_id)
          arc_zone_labels.append(None)
    arcs_matrix, arcs_labels, arc_ids = np.array(arcs_matrix), np.array(
      arcs_labels), np.array(arc_ids)
    arc_zone_labels = np.array(arc_zone_labels)
    np.save(data_set + '_arc_X', arcs_matrix)
    np.save(data_set + '_arc_Y', arcs_labels)
    np.save(data_set + '_arc_ids', arc_ids)
    np.save(data_set + '_arc_zone_labels', arc_zone_labels)


def _ground_truth_label(_id, zone):
  xml_base_zone_path = dp.REPO_HOME_PATH + '/data/bbox/aps_threats/zone_{2}/{0}_{1}.xml'
  for i in range(16):
    xml_zone_path = xml_base_zone_path.format(i, _id, zone)
    if os.path.exists(
        xml_zone_path) and len(parse_bbox_xml(xml_zone_path)) > 0:
      return 1
  return 0


def _curate_torso_calibration(data_set):
  ids = Table().read_table(dp.ids_by_data_set(data_set)).column('Id')
  X, y_label, y_zone, example_ids = [], [], [], []
  xml_base_zone_path = dp.REPO_HOME_PATH + '/data/bbox/aps_threats/zone_{2}/{0}_{1}.xml'
  for _id in ids:
    print(_id)
    logits, bboxes = threat_detect_inference(_id, 'torso')
    _, cropped_bbox = crop_and_pad(
      _id, file_format='aps', region='torso', slc=0, keypoint_pad=None)
    arcs = build_arcs('torso', logits, bboxes, _id)
    for i in range(len(arcs)):
      arc = arcs[i]
      matched = False
      for zone in [5, 6, 7, 17]:
        num_matches = 0
        original_scale_boxes = arc.original_scale_boxes(cropped_bbox)
        for i in range(original_scale_boxes.shape[0]):
          gt_xml_path = xml_base_zone_path.format(i, _id, zone)
          if os.path.exists(gt_xml_path):
            gt_bboxes = np.array(parse_bbox_xml(gt_xml_path))
            if gt_bboxes.size > 0:
              if np.max(
                  _iou(gt_bboxes, original_scale_boxes[i][np.newaxis])) > 0.5:
                num_matches += 1
        matched = num_matches / np.sum(
          np.sum(original_scale_boxes, axis=1) > 0) >= 0.5
        if matched:
          X.append(arc.to_list())
          y_label.append(int(matched))
          y_zone.append(zone)
          example_ids.append(_id)
          break
      if not matched:
        X.append(arc.to_list())
        y_label.append(int(matched))
        y_zone.append(zone)
        example_ids.append(_id)

  X, y_label, y_zone = np.array(X), np.array(y_label), np.array(y_zone)
  return X, y_label, y_zone, example_ids


def _curate_thigh_calibration(data_set):
  ids = Table().read_table(dp.ids_by_data_set(data_set)).column('Id')
  X, y_label, y_zone, example_ids = [], [], [], []
  xml_base_zone_path = dp.REPO_HOME_PATH + '/data/bbox/aps_threats/zone_{2}/{0}_{1}.xml'
  for _id in ids:
    print(_id)
    logits, bboxes = threat_detect_inference(_id, 'thigh')
    _, cropped_bbox = crop_and_pad(
      _id, file_format='aps', region='thigh', slc=0, keypoint_pad=None)
    arcs = build_arcs('thigh', logits, bboxes, _id)
    for i in range(len(arcs)):
      arc = arcs[i]
      matched = False
      for zone in [8, 9, 10, 11, 12]:
        num_matches = 0
        original_scale_boxes = arc.original_scale_boxes(cropped_bbox)
        for i in range(original_scale_boxes.shape[0]):
          gt_xml_path = xml_base_zone_path.format(i, _id, zone)
          if os.path.exists(gt_xml_path):
            gt_bboxes = np.array(parse_bbox_xml(gt_xml_path))
            if gt_bboxes.size > 0:
              num_matches += 1
        matched = num_matches / np.sum(
          np.sum(original_scale_boxes, axis=1) > 0) >= 0.5
        if matched:
          X.append(arc.to_list())
          y_label.append(int(matched))
          y_zone.append(zone)
          example_ids.append(_id)
          break
      if not matched:
        X.append(arc.to_list())
        y_label.append(int(matched))
        y_zone.append(zone)
        example_ids.append(_id)

  X, y_label, y_zone = np.array(X), np.array(y_label), np.array(y_zone)
  return X, y_label, y_zone, example_ids


def _curate_calf_calibration(data_set):
  ids = Table().read_table(dp.ids_by_data_set(data_set)).column('Id')
  X, y_label, y_zone = [], [], []

  for _id in ids:
    print(_id)
    logits, bboxes = threat_detect_inference(_id, 'calf')
    arcs = build_arcs('calf', logits, bboxes, _id)
    for i in range(len(arcs)):
      arc = arcs[i]
      X.append(arc.to_list())
      zone = i + 13
      y_label.append(_ground_truth_label(_id, zone))
      y_zone.append(zone)

  X, y_label, y_zone = np.array(X), np.array(y_label), np.array(y_zone)
  return X, y_label, y_zone, ids


def _curate_arm_calibration(data_set):
  ids = Table().read_table(dp.ids_by_data_set(data_set)).column('Id')
  X, y_label, y_zone = [], [], []

  for _id in ids:
    print(_id)
    logits, bboxes = threat_detect_inference(_id, 'arm')
    arcs = build_arcs('arm', logits, bboxes, _id)
    for i in range(len(arcs)):
      arc = arcs[i]
      X.append(arc.to_list())
      zone = i + 1
      y_label.append(_ground_truth_label(_id, zone))
      y_zone.append(zone)

  X, y_label, y_zone = np.array(X), np.array(y_label), np.array(y_zone)
  return X, y_label, y_zone, ids


def _curate_calibration(data_set, region):
  if region == 'arm':
    X, y_label, y_zone, ids = _curate_arm_calibration(data_set)
  if region == 'torso':
    X, y_label, y_zone, ids = _curate_torso_calibration(data_set)
  if region == 'thigh':
    X, y_label, y_zone, ids = _curate_thigh_calibration(data_set)
  if region == 'calf':
    X, y_label, y_zone, ids = _curate_calf_calibration(data_set)
  np.savez(
    dp.REPO_HOME_PATH + '/data/calibration/{0}_{1}'.format(data_set, region),
    X, y_label, y_zone, ids)


def load_calibration_data(data_set, region):
  outfile = np.load(dp.REPO_HOME_PATH + '/data/calibration/{0}_{1}.npz'.format(
    data_set, region))
  return (outfile['arr_{}'.format(i)] for i in range(4))


def main(argv=None):
  assert FLAGS.data_set and FLAGS.region
  _curate_calibration(FLAGS.data_set, FLAGS.region)


if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_string('data_set', None, 'data set to curate.')
  tf.app.flags.DEFINE_string('region', None, 'region to curate.')

  tf.app.run()