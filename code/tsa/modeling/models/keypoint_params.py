from collections import namedtuple
import numpy as np

FACE = namedtuple('keypoint', ['name', 'file_format', 'anchor_kwargs', 'slices'])
FACE.name = 'face'
FACE.file_format = 'a3daps'
FACE.anchor_kwargs = {
  'base_size': 10,
  'ratios': [1],
  'scales': 2**np.arange(3, 4),
  'allowed_border': 0
}
FACE.slices = [0]

BUTT = namedtuple('keypoint', ['name', 'file_format', 'anchor_kwargs', 'slices'])
BUTT.name = 'butt'
BUTT.file_format = 'aps'
BUTT.anchor_kwargs = {
  'base_size': 14,
  'ratios': [1, .5],
  'scales': 2**np.arange(3, 4),
  'allowed_border': 0
}
BUTT.slices = [8]