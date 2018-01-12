from tsa.modeling.models.keypoint_params import *
from tsa.utils.merge_bbox_xml import parse_bbox_xml
import tsa.utils.data_path as dp
import numpy as np
from PIL import Image


def crop_and_pad(_id, region, file_format, slc, keypoint_pad):
  if region == 'torso':
    if keypoint_pad is not None:
      cropped_image, cropped_bbox = _crop_torso(_id, file_format, slc,
                                                keypoint_pad)
    else:
      cropped_image, cropped_bbox = _crop_torso(_id, file_format, slc)
  if region == 'thigh':
    if keypoint_pad is not None:
      cropped_image, cropped_bbox = _crop_thigh(_id, file_format, slc,
                                                keypoint_pad)
    else:
      cropped_image, cropped_bbox = _crop_thigh(_id, file_format, slc)
  if region == 'calf':
    if keypoint_pad is not None:
      cropped_image, cropped_bbox = _crop_calf(_id, file_format, slc,
                                               keypoint_pad)
    else:
      cropped_image, cropped_bbox = _crop_calf(_id, file_format, slc)
  if region == 'arm':
    if keypoint_pad is not None:
      cropped_image, cropped_bbox = _crop_arm(_id, file_format, slc,
                                              keypoint_pad)
    else:
      cropped_image, cropped_bbox = _crop_arm(_id, file_format, slc)
  return zero_pad_image(cropped_image).astype(np.float64), cropped_bbox


def _crop_torso(_id, file_format, slc, keypoint_pad=25, include_face=False):
  face_bbox = _get_keypoint_bbox(_id, FACE)
  butt_bbox = _get_keypoint_bbox(_id, BUTT)
  if include_face:
    torso_y_min = face_bbox[1] - keypoint_pad
  else:
    torso_y_min = face_bbox[3] - keypoint_pad
  torso_y_max = butt_bbox[1] + keypoint_pad

  image = _get_image(_id, file_format, slc)
  image = image[torso_y_min:torso_y_max, :]

  x_min, x_max = 131, 381

  cropped_bbox = [x_min, torso_y_min, x_max, torso_y_max]
  return image[:, x_min:x_max], cropped_bbox


def _crop_thigh(_id, file_format, slc, keypoint_pad=30):
  butt_bbox = _get_keypoint_bbox(_id, BUTT)
  y_min = butt_bbox[1]
  y_max = butt_bbox[1] + (660 - butt_bbox[1]) // 2 + keypoint_pad

  image = _get_image(_id, file_format, slc)
  image = image[y_min:y_max, :]

  inds = np.tile(np.arange(512), y_max - y_min)
  try:
    x_center = int(np.average(inds, weights=image.flatten()))
  except:
    x_center = 512 // 2
  x_min, x_max = max(x_center - 100, 0), min(x_center + 100, 511)

  cropped_bbox = [x_min, y_min, x_max, y_max]
  return image[:, x_min:x_max], cropped_bbox


def _crop_calf(_id, file_format, slc, keypoint_pad=0):
  butt_bbox = _get_keypoint_bbox(_id, BUTT)
  y_min = butt_bbox[1] + (660 - butt_bbox[1]) // 2 - keypoint_pad
  y_max = 660

  image = _get_image(_id, file_format, slc)
  image = image[y_min:y_max, :]

  inds = np.tile(np.arange(512), y_max - y_min)
  try:
    x_center = int(np.average(inds, weights=image.flatten()))
  except:
    x_center = 512 // 2
  x_min, x_max = max(x_center - 150, 0), min(x_center + 150, 511)

  cropped_bbox = [x_min, y_min, x_max, y_max]
  return image[:, x_min:x_max], cropped_bbox


def _crop_arm(_id, file_format, slc, keypoint_pad=30):
  face_bbox = _get_keypoint_bbox(_id, FACE)
  y_min = 50
  y_max = face_bbox[3] + keypoint_pad

  image = _get_image(_id, file_format, slc)
  image = image[y_min:y_max, :]

  inds = np.tile(np.arange(512), y_max - y_min)
  x_min, x_max = 50, 511 - 50

  cropped_bbox = [x_min, y_min, x_max, y_max]
  return image[:, x_min:x_max], cropped_bbox


def _get_keypoint_bbox(_id, keypoint):
  return parse_bbox_xml(dp.REPO_HOME_PATH +
                        '/data/bbox/keypoint_inference/{0}/{1}.xml'.format(
                          keypoint.name, _id))[0]


def _get_image(_id, file_format, slc):
  base_png_path = dp.LARGE_DATA_BIN + '/data/raw/'
  if file_format == 'a3daps':
    base_png_path += 'png/'
  else:
    base_png_path += 'aps_png/'
  return np.array(
    Image.open('{0}/{1}/{1}_{2}.png'.format(base_png_path, slc, _id))).astype(
      np.float32)


def zero_pad_image(image):
  height, width = image.shape
  if height == width:
    return image
  if height > width:
    image = np.transpose(image)
  image = vertical_pad_image(image)
  if height > width:
    image = np.transpose(image)
  return image


def vertical_pad_image(image):
  height, width = image.shape
  assert width > height
  top_pad = (width - height) // 2
  bottom_pad = width - height - top_pad
  top_zeros = np.zeros((top_pad, width))
  bottom_zeros = np.zeros((bottom_pad, width))
  return np.vstack((top_zeros, image, bottom_zeros))


def pad_dim(image_dim):
  height, width = image_dim
  pad = (max(height, width) - min(height, width)) // 2
  if height > width:
    return (0, pad)
  return (pad, 0)