import numpy as np


def vertical_stripe_mask(image_dim, dot_dim, stride_dim):
  zeros_dot = np.zeros(dot_dim)
  ones_dot = np.ones(stride_dim)
  mask = np.hstack((zeros_dot, ones_dot))
  mask = np.tile(mask, image_dim[1] // mask.shape[1])
  mask = np.tile(mask, [image_dim[0] // mask.shape[0], 1])
  image_ones = np.ones(image_dim)
  image_ones[:mask.shape[0], :mask.shape[1]] = mask
  return image_ones


def horizontal_stripe_mask(image_dim, dot_dim, stride_dim):
  zeros_dot = np.zeros(dot_dim)
  ones_dot = np.ones(stride_dim)
  mask = np.vstack((zeros_dot, ones_dot))
  mask = np.tile(mask, image_dim[1] // mask.shape[1])
  mask = np.tile(mask, [image_dim[0] // mask.shape[0], 1])
  image_ones = np.ones(image_dim)
  image_ones[:mask.shape[0], :mask.shape[1]] = mask
  return image_ones


def squares_mask(image_dim, dot_dim, stride_dim):
  zeros_dot = np.zeros(dot_dim)
  ones_dot = np.ones((dot_dim[0], stride_dim[1]))
  row_1_patch = np.hstack((zeros_dot, ones_dot))
  row_1 = np.tile(row_1_patch, image_dim[1] // row_1_patch.shape[1])
  row_1 = np.vstack((row_1, np.ones((stride_dim[0], row_1.shape[1]))))
  row_2_patch = np.hstack((ones_dot, zeros_dot))
  row_2 = np.tile(row_2_patch, image_dim[1] // row_2_patch.shape[1])
  row_2 = np.vstack((row_2, np.ones((stride_dim[0], row_2.shape[1]))))
  row = np.vstack((row_1, row_2))
  mask = np.tile(row, [image_dim[0] // row.shape[0], 1])
  image_ones = np.ones(image_dim)
  image_ones[:mask.shape[0], :mask.shape[1]] = mask
  return image_ones


def make_vertical_stripe_mask(dot_dim, stride_dim):
  def mask(image_dim):
    return vertical_stripe_mask(image_dim, dot_dim, stride_dim)

  return mask


def make_horizontal_stripe_mask(dot_dim, stride_dim):
  def mask(image_dim):
    return horizontal_stripe_mask(image_dim, dot_dim, stride_dim)

  return mask


def make_squares_mask(dot_dim, stride_dim):
  def mask(image_dim):
    return squares_mask(image_dim, dot_dim, stride_dim)

  return mask
