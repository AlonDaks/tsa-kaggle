from datascience import *
import numpy as np
import tensorflow as tf
from tsa.modeling.models.localization_net import LocalizationModel
from tsa.modeling.models.keypoint_crop import crop_and_pad
import tsa.utils.data_path as dp
from skimage.transform import resize

TRAIN_DIR = {
  'arm': dp.train_dir('arm_localization'),
  'torso': dp.train_dir('torso_localization'),
  'thigh': dp.train_dir('thigh_localization'),
  'calf': dp.train_dir('calf_localization')
}

MEAN_SUBTRACT = {'arm': 5.5, 'torso': 23.4, 'thigh': 35.0, 'calf': 8.5}

IMAGE_LENGTH = 200

REGION_ZONES = {'arm': [1, 2, 3, 4], 'thigh': [8, 9, 10, 11, 12]}

BATCH_SIZE = 1

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


def _region_localization_inference(region):
  with REGION_GRAPHS[region].as_default():
    images_ph = tf.placeholder(tf.float32, (None, IMAGE_LENGTH, IMAGE_LENGTH,
                                            1))
    images_ph -= MEAN_SUBTRACT[region]
    slcs_ph = tf.placeholder(tf.int64, (None, ))

    model = LocalizationModel(region, BATCH_SIZE)
    logits_tensor = model.inference(images_ph, slcs_ph, test=True)
    logits_tensor = tf.nn.softmax(logits_tensor)

    saver = tf.train.Saver()
    saver.restore(
      REGION_SESSIONS[region],
      tf.train.get_checkpoint_state(TRAIN_DIR[region]).model_checkpoint_path)

    return images_ph, slcs_ph, logits_tensor


REGION_TENSORS = {
  'arm': _region_localization_inference('arm'),
  'thigh': _region_localization_inference('thigh')
}


def localization_inference(_id, region, bbox, slc):
  images_ph, slcs_ph, logits_tensor = REGION_TENSORS[region]

  image = crop_and_pad(_id, region, 'aps', slc, keypoint_pad=None)[0]
  image = resize(image, (IMAGE_LENGTH, IMAGE_LENGTH), mode='constant')
  image[bbox[1]:bbox[3], bbox[0]:bbox[2]] += 150
  image = np.minimum(image, 255)
  images = image[np.newaxis, :, :, np.newaxis]

  logits = REGION_SESSIONS[region].run(
    [logits_tensor], feed_dict={images_ph: images, slcs_ph: np.array([slc])})
  return REGION_ZONES[region][np.argmax(logits)]