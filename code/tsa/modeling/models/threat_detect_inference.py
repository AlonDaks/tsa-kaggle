from datascience import *
import numpy as np
import tensorflow as tf
from tsa.modeling.models.threat_detect_net import ThreatDetectModel, target_to_box
from tsa.modeling.models.keypoint_crop import crop_and_pad
import tsa.utils.data_path as dp
from skimage.transform import resize

TRAIN_DIR = {
  'arm': dp.train_dir('arm_full'),
  'torso': dp.train_dir('torso_full'),
  'thigh': dp.train_dir('thigh_full'),
  'calf': dp.train_dir('calf_full')
}

BASE_SIZES = {
  'arm': [64, 128],
  'torso': [128, 256],
  'thigh': [128, 256],
  'calf': [128, 128]
}

MEAN_SUBTRACT = {'arm': 5.5, 'torso': 23.4, 'thigh': 35.0, 'calf': 5.5}

BATCH_SIZE = 16

REGION_GRAPHS = {
  'arm': tf.Graph(),
  'torso': tf.Graph(),
  'thigh': tf.Graph(),
  'calf': tf.Graph()
}

REGION_SESSIONS = {
  'arm': tf.Session(graph=REGION_GRAPHS['arm']),
  'torso': tf.Session(graph=REGION_GRAPHS['torso']),
  'thigh': tf.Session(graph=REGION_GRAPHS['thigh']),
  'calf': tf.Session(graph=REGION_GRAPHS['calf'])
}


def _region_threat_detect_inference(region):
  with REGION_GRAPHS[region].as_default():
    images_ph = tf.placeholder(tf.float32, (None, 600, 600, 1))
    images_ph -= MEAN_SUBTRACT[region]

    model = ThreatDetectModel(
      batch_size=BATCH_SIZE,
      image_dim=(600, 600),
      base_sizes=BASE_SIZES[region])
    logits_tensor, bbox_tensor = model.inference(images_ph, test=True)
    logits_tensor = tf.sigmoid(logits_tensor)
    bbox_tensor = target_to_box(bbox_tensor, model.anchors)

    saver = tf.train.Saver()
    saver.restore(
      REGION_SESSIONS[region],
      tf.train.get_checkpoint_state(TRAIN_DIR[region]).model_checkpoint_path)

    return images_ph, logits_tensor, bbox_tensor


REGION_TENSORS = {
  'arm': _region_threat_detect_inference('arm'),
  'torso': _region_threat_detect_inference('torso'),
  'thigh': _region_threat_detect_inference('thigh'),
  'calf': _region_threat_detect_inference('calf'),
}


def threat_detect_inference(_id, region):
  images_ph, logits_tensor, bbox_tensor = REGION_TENSORS[region]

  images = [
    crop_and_pad(_id, region, 'aps', slc=i, keypoint_pad=None)[0]
    for i in range(BATCH_SIZE)
  ]
  images = [resize(i, (600, 600), mode='constant') for i in images]
  images = np.concatenate([i[np.newaxis, :, :] for i in images])
  images = images[:, :, :, np.newaxis]

  logits, bbox = REGION_SESSIONS[region].run(
    [logits_tensor, bbox_tensor], feed_dict={images_ph: images})
  return logits, bbox