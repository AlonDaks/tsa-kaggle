import tensorflow as tf
from tsa.modeling.models.keypoint_crop import *
from tsa.modeling.models.keypoint_params import FACE, BUTT
import tsa.utils.data_path as dp
import tsa.utils.constants as CONST
from tsa.utils.masks import *
from tsa.utils.merge_bbox_xml import parse_bbox_xml
import numpy as np
import os
from PIL import Image
from datascience import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_set', None, """data_set""")
tf.app.flags.DEFINE_string('file_format', None, """file_format""")
tf.app.flags.DEFINE_string('region', None, """region""")
tf.app.flags.DEFINE_integer('keypoint_pad', None, """keypoint_pad""")
tf.app.flags.DEFINE_integer('random_seed', 29, """keypoint_pad""")
tf.app.flags.DEFINE_string('mode', 'full', """full, mask, or pure""")

BOX_COUNT = 5


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bbox_feature(bboxes):
  assert len(bboxes) % 4 == 0, 'bboxes len must be multiple of 4'
  return tf.train.Feature(int64_list=tf.train.Int64List(value=bboxes))


def _create_full_tf_record(data_set,
                           file_format,
                           region,
                           keypoint_pad,
                           random_seed,
                           example_multiple=3):
  np.random.seed(random_seed)

  ids = Table().read_table(dp.ids_by_data_set(data_set)).column('Id')

  tf_record_name = '{0}_crop_patch_full_{1}'.format(file_format, region)
  print('Creating: {0}_{1}'.format(data_set, tf_record_name))
  writer = tf.python_io.TFRecordWriter(dp.tf_records(data_set, tf_record_name))

  base_xml_path = '{0}/data/bbox/{1}_merged_threats_{2}'.format(
    dp.REPO_HOME_PATH, file_format, region)

  masks = [
    make_vertical_stripe_mask((4, 4), (4, 4)),
    make_vertical_stripe_mask((4, 4), (4, 15)),
    make_horizontal_stripe_mask((4, 4), (4, 4)),
    make_horizontal_stripe_mask((4, 4), (15, 4)),
    make_squares_mask((10, 10), (10, 25)),
    make_squares_mask((5, 5), (10, 25)),
    make_squares_mask((5, 5), (5, 15)),
    make_vertical_stripe_mask((2, 2), (2, 2))
  ]

  if file_format == 'aps':
    slcs = range(16)
  if file_format == 'a3daps':
    slcs = [t for t in range(64) if t % 2 == 0]

  existing_paths, non_existing_paths = 0, 0
  for _id in ids:
    for i in slcs:
      xml_path = '{0}/{1}_{2}.xml'.format(base_xml_path, i, _id)
      if os.path.exists(xml_path):
        existing_paths += 1
      else:
        non_existing_paths += 1

  ids_and_slcs = []
  for _id in ids:
    for i in slcs:
      xml_path = '{0}/{1}_{2}.xml'.format(base_xml_path, i, _id)
      if os.path.exists(xml_path):
        ids_and_slcs += example_multiple * [(_id, i)]
      elif np.random.binomial(
          1, min(1, example_multiple * existing_paths / non_existing_paths)):
        ids_and_slcs += [((_id, i))]
  np.random.shuffle(ids_and_slcs)
  for _id, i in ids_and_slcs:
    image, cropped_bbox = crop_and_pad(
      _id, region, file_format, i, keypoint_pad=keypoint_pad)
    pad = pad_dim((cropped_bbox[3] - cropped_bbox[1],
                   cropped_bbox[2] - cropped_bbox[0]))
    xml_path = '{0}/{1}_{2}.xml'.format(base_xml_path, i, _id)

    if os.path.exists(xml_path):
      bboxes = parse_bbox_xml(xml_path)
      if np.sum(np.array(bboxes).ravel()) == 0:
        continue
      cropped_bboxes = []
      for box in bboxes:
        rescaled_box = [
          box[0] - cropped_bbox[0] + pad[1], box[1] - cropped_bbox[1] + pad[0],
          box[2] - cropped_bbox[0] + pad[1], box[3] - cropped_bbox[1] + pad[0]
        ]
        rescaled_box[0] = max(rescaled_box[0], pad[1])
        rescaled_box[1] = max(rescaled_box[1], pad[0])
        rescaled_box[2] = min(rescaled_box[2],
                              cropped_bbox[2] - cropped_bbox[0] + pad[1])
        rescaled_box[3] = min(rescaled_box[3],
                              cropped_bbox[3] - cropped_bbox[1] + pad[0])
        cropped_bboxes.append(rescaled_box)
      bboxes = np.array(cropped_bboxes).ravel()
      bboxes = np.append(bboxes,
                         (BOX_COUNT * 4 - len(bboxes)) * [0]).astype(np.int64)
    else:
      bboxes = np.array(BOX_COUNT * 4 * [0]).astype(np.int64)

    image = image.astype(np.float32)
    example = tf.train.Example(features=tf.train.Features(
      feature={
        'image':
        _bytes_feature(image.tostring()),
        'bbox':
        _bbox_feature(bboxes),
        'dim':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[0]]))
      }))
    writer.write(example.SerializeToString())
    if 'train' in data_set:
      for _ in range(len(masks) - 1):
        writer.write(example.SerializeToString())
      for mask in masks:
        masked_image = (image * mask(image.shape)).astype(np.float32)
        example = tf.train.Example(features=tf.train.Features(
          feature={
            'image':
            _bytes_feature(masked_image.tostring()),
            'bbox':
            _bbox_feature(bboxes),
            'dim':
            tf.train.Feature(int64_list=tf.train.Int64List(
              value=[image.shape[0]]))
          }))
        writer.write(example.SerializeToString())
  writer.close()


def _create_mask_tf_record(data_set,
                           file_format,
                           region,
                           keypoint_pad,
                           random_seed,
                           example_multiple=3):
  np.random.seed(random_seed)

  ids = Table().read_table(dp.ids_by_data_set(data_set)).column('Id')

  tf_record_name = '{0}_crop_patch_mask_{1}'.format(file_format, region)
  print('Creating: {0}_{1}'.format(data_set, tf_record_name))
  writer = tf.python_io.TFRecordWriter(dp.tf_records(data_set, tf_record_name))

  base_xml_path = '{0}/data/bbox/{1}_merged_threats_{2}'.format(
    dp.REPO_HOME_PATH, file_format, region)

  masks = [
    make_vertical_stripe_mask((4, 4), (4, 4)),
    make_vertical_stripe_mask((4, 4), (4, 15)),
    make_horizontal_stripe_mask((4, 4), (4, 4)),
    make_horizontal_stripe_mask((4, 4), (15, 4)),
    make_squares_mask((10, 10), (10, 25)),
    make_squares_mask((5, 5), (10, 25)),
    make_squares_mask((5, 5), (5, 15)),
    make_vertical_stripe_mask((2, 2), (2, 2))
  ]

  if file_format == 'aps':
    slcs = range(16)
  if file_format == 'a3daps':
    slcs = [t for t in range(64) if t % 2 == 0]

  ids_and_slcs = []
  for _id in ids:
    for i in slcs:
      xml_path = '{0}/{1}_{2}.xml'.format(base_xml_path, i, _id)
      if os.path.exists(xml_path):
        ids_and_slcs += [(_id, i)]

  np.random.shuffle(ids_and_slcs)

  for _id, i in ids_and_slcs:
    image, cropped_bbox = crop_and_pad(
      _id, region, file_format, i, keypoint_pad=keypoint_pad)
    pad = pad_dim((cropped_bbox[3] - cropped_bbox[1],
                   cropped_bbox[2] - cropped_bbox[0]))
    xml_path = '{0}/{1}_{2}.xml'.format(base_xml_path, i, _id)

    if os.path.exists(xml_path):
      bboxes = parse_bbox_xml(xml_path)
      if np.sum(np.array(bboxes).ravel()) == 0:
        continue
      cropped_bboxes = []
      for box in bboxes:
        rescaled_box = [
          box[0] - cropped_bbox[0] + pad[1], box[1] - cropped_bbox[1] + pad[0],
          box[2] - cropped_bbox[0] + pad[1], box[3] - cropped_bbox[1] + pad[0]
        ]
        rescaled_box[0] = max(rescaled_box[0], pad[1])
        rescaled_box[1] = max(rescaled_box[1], pad[0])
        rescaled_box[2] = min(rescaled_box[2],
                              cropped_bbox[2] - cropped_bbox[0] + pad[1])
        rescaled_box[3] = min(rescaled_box[3],
                              cropped_bbox[3] - cropped_bbox[1] + pad[0])
        cropped_bboxes.append(rescaled_box)
      bboxes = np.array(cropped_bboxes).ravel()
      bboxes = np.append(bboxes,
                         (BOX_COUNT * 4 - len(bboxes)) * [0]).astype(np.int64)
    else:
      continue

    image = image.astype(np.float32)
    example = tf.train.Example(features=tf.train.Features(
      feature={
        'image':
        _bytes_feature(image.tostring()),
        'bbox':
        _bbox_feature(bboxes),
        'dim':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[0]]))
      }))
    writer.write(example.SerializeToString())
    if 'train' in data_set:
      for _ in range(len(masks) - 1):
        writer.write(example.SerializeToString())
      for mask in masks:
        masked_image = (image * mask(image.shape)).astype(np.float32)
        example = tf.train.Example(features=tf.train.Features(
          feature={
            'image':
            _bytes_feature(masked_image.tostring()),
            'bbox':
            _bbox_feature(bboxes),
            'dim':
            tf.train.Feature(int64_list=tf.train.Int64List(
              value=[image.shape[0]]))
          }))
        writer.write(example.SerializeToString())
  writer.close()


def _create_pure_tf_record(data_set,
                           file_format,
                           region,
                           keypoint_pad,
                           random_seed,
                           example_multiple=3):
  np.random.seed(random_seed)

  ids = Table().read_table(dp.ids_by_data_set(data_set)).column('Id')

  tf_record_name = '{0}_crop_patch_pure_{1}'.format(file_format, region)
  print('Creating: {0}_{1}'.format(data_set, tf_record_name))
  writer = tf.python_io.TFRecordWriter(dp.tf_records(data_set, tf_record_name))

  base_xml_path = '{0}/data/bbox/{1}_merged_threats_{2}'.format(
    dp.REPO_HOME_PATH, file_format, region)

  if file_format == 'aps':
    slcs = range(16)
  if file_format == 'a3daps':
    slcs = [t for t in range(64) if t % 2 == 0]

  ids_and_slcs = []
  for _id in ids:
    for i in slcs:
      xml_path = '{0}/{1}_{2}.xml'.format(base_xml_path, i, _id)
      if os.path.exists(xml_path):
        ids_and_slcs += [(_id, i)]

  np.random.shuffle(ids_and_slcs)

  for _id, i in ids_and_slcs:
    image, cropped_bbox = crop_and_pad(
      _id, region, file_format, i, keypoint_pad=keypoint_pad)
    pad = pad_dim((cropped_bbox[3] - cropped_bbox[1],
                   cropped_bbox[2] - cropped_bbox[0]))
    xml_path = '{0}/{1}_{2}.xml'.format(base_xml_path, i, _id)

    if os.path.exists(xml_path):
      bboxes = parse_bbox_xml(xml_path)
      if np.sum(np.array(bboxes).ravel()) == 0:
        continue
      cropped_bboxes = []
      for box in bboxes:
        rescaled_box = [
          box[0] - cropped_bbox[0] + pad[1], box[1] - cropped_bbox[1] + pad[0],
          box[2] - cropped_bbox[0] + pad[1], box[3] - cropped_bbox[1] + pad[0]
        ]
        rescaled_box[0] = max(rescaled_box[0], pad[1])
        rescaled_box[1] = max(rescaled_box[1], pad[0])
        rescaled_box[2] = min(rescaled_box[2],
                              cropped_bbox[2] - cropped_bbox[0] + pad[1])
        rescaled_box[3] = min(rescaled_box[3],
                              cropped_bbox[3] - cropped_bbox[1] + pad[0])
        cropped_bboxes.append(rescaled_box)
      bboxes = np.array(cropped_bboxes).ravel()
      bboxes = np.append(bboxes,
                         (BOX_COUNT * 4 - len(bboxes)) * [0]).astype(np.int64)
    else:
      continue

    image = image.astype(np.float32)
    example = tf.train.Example(features=tf.train.Features(
      feature={
        'image':
        _bytes_feature(image.tostring()),
        'bbox':
        _bbox_feature(bboxes),
        'dim':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[0]]))
      }))
    writer.write(example.SerializeToString())
  writer.close()


def _create_localization_tf_record(data_set, region, random_seed):
  np.random.seed(random_seed)

  ids = Table().read_table(dp.ids_by_data_set(data_set)).column('Id')

  tf_record_name = '{0}_localization'.format(region)
  print('Creating: {0}_{1}'.format(data_set, tf_record_name))
  writer = tf.python_io.TFRecordWriter(dp.tf_records(data_set, tf_record_name))

  base_xml_path = '{0}/data/bbox/aps_threats/zone_'.format(dp.REPO_HOME_PATH)

  if region == 'arm':
    zones = {1: 0, 2: 1, 3: 2, 4: 3}
  if region == 'thigh':
    zones = {8: 0, 9: 1, 10: 2, 11: 3, 12: 4}

  masks = [
    make_vertical_stripe_mask((4, 4), (4, 4)),
    make_vertical_stripe_mask((4, 4), (4, 15)),
    make_horizontal_stripe_mask((4, 4), (4, 4)),
    make_horizontal_stripe_mask((4, 4), (15, 4)),
    make_squares_mask((10, 10), (10, 25)),
    make_squares_mask((5, 5), (10, 25)),
    make_squares_mask((5, 5), (5, 15)),
    make_vertical_stripe_mask((2, 2), (2, 2))
  ]

  examples = []
  for _id in ids:
    for slc in range(16):
      for zone in zones.keys():
        xml_path = base_xml_path + '{0}/{1}_{2}.xml'.format(zone, slc, _id)
        if os.path.exists(xml_path):
          boxes = parse_bbox_xml(xml_path)
          if len(boxes) > 0:
            examples += [(_id, slc, zone)]

  np.random.shuffle(examples)

  for _id, slc, zone in examples:
    image, cropped_bbox = crop_and_pad(
      _id, region, 'aps', slc, keypoint_pad=None)
    pad = pad_dim((cropped_bbox[3] - cropped_bbox[1],
                   cropped_bbox[2] - cropped_bbox[0]))
    xml_path = base_xml_path + '{0}/{1}_{2}.xml'.format(zone, slc, _id)

    box = parse_bbox_xml(xml_path)[0]
    rescaled_box = [
      box[0] - cropped_bbox[0] + pad[1], box[1] - cropped_bbox[1] + pad[0],
      box[2] - cropped_bbox[0] + pad[1], box[3] - cropped_bbox[1] + pad[0]
    ]
    rescaled_box[0] = max(rescaled_box[0], pad[1])
    rescaled_box[1] = max(rescaled_box[1], pad[0])
    rescaled_box[2] = min(rescaled_box[2],
                          cropped_bbox[2] - cropped_bbox[0] + pad[1])
    rescaled_box[3] = min(rescaled_box[3],
                          cropped_bbox[3] - cropped_bbox[1] + pad[0])
    bboxes = np.array(rescaled_box).astype(np.int64)
    image = image.astype(np.float32)
    label = zones[zone]
    example = tf.train.Example(features=tf.train.Features(
      feature={
        'image':
        _bytes_feature(image.tostring()),
        'bbox':
        _bbox_feature(bboxes),
        'dim':
        tf.train.Feature(int64_list=tf.train.Int64List(
          value=[image.shape[0]])),
        'label':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
        'slc':
        tf.train.Feature(int64_list=tf.train.Int64List(value=[slc]))
      }))
    writer.write(example.SerializeToString())
    if 'train' in data_set:
      for _ in range(len(masks) - 1):
        writer.write(example.SerializeToString())
      for mask in masks:
        masked_image = (image * mask(image.shape)).astype(np.float32)
        example = tf.train.Example(features=tf.train.Features(
          feature={
            'image':
            _bytes_feature(masked_image.tostring()),
            'bbox':
            _bbox_feature(bboxes),
            'dim':
            tf.train.Feature(int64_list=tf.train.Int64List(
              value=[image.shape[0]])),
            'label':
            tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'slc':
            tf.train.Feature(int64_list=tf.train.Int64List(value=[slc]))
          }))
        writer.write(example.SerializeToString())
  writer.close()


def create_keypoint_tf_record(data_set, region, random_seed):
  np.random.seed(random_seed)

  if region == 'face':
    keypoint = FACE
  elif region == 'butt':
    keypoint = BUTT

  ids = Table().read_table(dp.ids_by_data_set(data_set)).column('Id')

  tf_record_name = '{}_keypoint'.format(keypoint.name)
  print('Creating: {0}_{1}'.format(data_set, tf_record_name))
  writer = tf.python_io.TFRecordWriter(dp.tf_records(data_set, tf_record_name))

  base_xml_path = dp.REPO_HOME_PATH + '/data/bbox/' + keypoint.name
  base_png_path = dp.LARGE_DATA_BIN + '/data/raw/'
  if keypoint.file_format == 'a3daps':
    base_png_path += 'a3daps_png/'
  else:
    base_png_path += 'aps_png/'
  for _id in ids:
    for i in keypoint.slices:
      image = np.array(
        Image.open('{0}/{1}/{1}_{2}.png'.format(base_png_path, i,
                                                _id))).astype(np.float32)
      xml_path = '{0}/{1}_{2}.xml'.format(base_xml_path, i, _id)
      if os.path.exists(xml_path):
        bboxes = np.array(parse_bbox_xml(xml_path)).ravel()
        bboxes = np.append(
          bboxes, (BOX_COUNT * 4 - len(bboxes)) * [0]).astype(np.int64)
        if np.sum(bboxes) == 0:
          continue
        example = tf.train.Example(features=tf.train.Features(
          feature={
            'image': _bytes_feature(image.tostring()),
            'bbox': _bbox_feature(bboxes),
          }))
        writer.write(example.SerializeToString())
  writer.close()


def _create_tf_record(data_set, file_format, region, keypoint_pad, random_seed,
                      mode):
  if mode == 'full':
    _create_full_tf_record(data_set, file_format, region, keypoint_pad,
                           random_seed)
  if mode == 'mask':
    _create_mask_tf_record(data_set, file_format, region, keypoint_pad,
                           random_seed)
  if mode == 'pure':
    _create_pure_tf_record(data_set, file_format, region, keypoint_pad,
                           random_seed)
  if mode == 'localization':
    _create_localization_tf_record(data_set, region, random_seed)
  if mode == 'keypoint':
    create_keypoint_tf_record(data_set, region, random_seed)


def main(argv=None):
  _create_tf_record(FLAGS.data_set, FLAGS.file_format, FLAGS.region,
                    FLAGS.keypoint_pad, FLAGS.random_seed, FLAGS.mode)


if __name__ == '__main__':
  tf.app.run()