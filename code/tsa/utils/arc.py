import numpy as np
from tsa.utils.non_max_suppression import non_max_suppression
from tsa.modeling.models.keypoint_crop import pad_dim, crop_and_pad
from sklearn.cluster import KMeans
from skimage.transform import resize
from tsa.modeling.models.localization_inference import localization_inference


class Arc(object):
  def __init__(self, num_slcs=16):
    self.num_slcs = num_slcs
    self.arc = num_slcs * [None]
    self.start_slc = None

  def __len__(self):
    length = 0
    for elem in self.arc:
      if elem is not None:
        length += 1
    return length

  def __iadd__(self, other):
    assert self.is_disjoint_with(other)
    for i in range(other.num_slcs):
      if other[i]:
        self[i] = other[i]
    return self

  def __getitem__(self, key):
    return self.arc[key]

  def __setitem__(self, key, value):
    assert self.arc[key] is None
    if self.start_slc is None:
      self.start_slc = key
    self.arc[key] = value

  def __delitem__(self, key):
    self.arc[key] = None

  def __str__(self):
    return '\n'.join(
      ['Slice {0}: {1}'.format(i, self[i]) for i in range(self.num_slcs)])

  def __eq__(self, other):
    assert self.num_slcs == other.num_slcs
    for i in range(self.num_slcs):
      if self[i] and not other[i] or not self[i] and other[i]:
        return False
      if self[i] is None and other[i] is None:
        continue
      if not np.array_equal(self[i][0],
                            other[i][0]) or self[i][1] != other[i][1]:
        return False
    return True

  def __hash__(self):
    return hash(str(self))

  def mean_y_std(self):
    y_mins, y_maxes = [], []
    for i in range(self.num_slcs):
      if self[i]:
        y_mins.append(self[i][0][1])
        y_maxes.append(self[i][0][3])
    return (np.std(y_mins) + np.std(y_maxes)) / 2

  def total_prob(self):
    total_prob = 0
    for i in range(self.num_slcs):
      if self[i]:
        total_prob += self[i][1]
    return total_prob

  def boxes(self):
    boxes = []
    for i in range(self.num_slcs):
      if self[i]:
        boxes.append(self[i][0])
    return np.array(boxes)

  def boxes_through_index(self, index):
    boxes = []
    for i in np.arange(self.start_slc,
                       self.start_slc + index + 1) % self.num_slcs:
      if self[i]:
        boxes.append(self[i][0])
    return np.array(boxes)

  def split(self, new_start_slc):
    new_arc = Arc()
    for i in np.arange(new_start_slc,
                       new_start_slc + self.num_slcs) % self.num_slcs:
      if i == self.start_slc:
        break
      new_arc[i] = self[i]
      del self[i]
    return new_arc

  def is_disjoint_with(self, other):
    for i in range(self.num_slcs):
      if self[i] and other[i]:
        return False
    return True

  def can_be_merged_with(self, other, max_gap, y_iou_thresh):
    if not self.is_disjoint_with(other):
      return False

    self_start_slc = self.start_slc
    self_end_slc = self.end_slc
    other_start_slc = other.start_slc
    other_end_slc = other.end_slc

    gaps = gaps_between(self_start_slc, self_end_slc, other_start_slc,
                        other_end_slc, self.num_slcs)
    if not min(gaps) <= max_gap:
      return False
    if np.min(y_iou(self.boxes(), other.boxes())) < y_iou_thresh:
      return False
    if gaps[0] <= max_gap:
      start_box = self[self_start_slc][0]
      end_box = other[other_end_slc][0]
      if start_box[0] > (end_box[0] + end_box[2]) / 2:
        return False
    if gaps[1] <= max_gap:
      start_box = other[other_start_slc][0]
      end_box = self[self_end_slc][0]
      if start_box[0] > (end_box[0] + end_box[2]) / 2:
        return False
    return True

  @property
  def end_slc(self):
    inds = (np.arange(self.start_slc, self.start_slc + self.num_slcs) %
            self.num_slcs)[::-1]
    for i in inds:
      if self[i]:
        return i

  def original_scale_boxes(self, cropped_bbox):
    pads = pad_dim((cropped_bbox[3] - cropped_bbox[1],
                    cropped_bbox[2] - cropped_bbox[0]))
    boxes = []
    for i in range(self.num_slcs):
      if self[i]:
        boxes.append(self[i][0])
      else:
        boxes.append(4 * [0])
    boxes = np.array(boxes)
    original_size = max(2 * pads[0] + cropped_bbox[3] - cropped_bbox[1],
                        2 * pads[1] + cropped_bbox[2] - cropped_bbox[0])
    original_boxes = boxes * original_size / 600
    original_boxes[:, [0, 2]] -= pads[1]
    original_boxes[:, [1, 3]] -= pads[0]
    original_boxes[:, [0, 2]] += cropped_bbox[0]
    original_boxes[:, [1, 3]] += cropped_bbox[1]
    return (np.sum(boxes, axis=1) > 0)[:, np.newaxis] * original_boxes

  def to_list(self):
    array = []
    for i in range(self.num_slcs):
      if self[i]:
        array += self[i][0].tolist()
        array.append(self[i][1])
      else:
        array += 5 * [0]
    return array

  def _torso_calibration_featurize(self):
    X = np.array(self.to_list())[np.newaxis]
    probs = X[:, 4::5]
    arc_length = np.sum(probs > 0, axis=1)[:, np.newaxis]
    probs_sum = np.sum(probs, axis=1)[:, np.newaxis]
    y_mins, y_maxes = X[:, 1::5], X[:, 3::5]
    mean_y_min = np.average(y_mins, axis=1, weights=y_mins != 0)
    mean_y_max = np.average(y_maxes, axis=1, weights=y_maxes != 0)
    mean_y = ((mean_y_min + mean_y_max) / 2)[:, np.newaxis]
    max_prob = np.max(probs, axis=1)[:, np.newaxis]
    probs_mean = probs_sum / arc_length
    log_probs = np.log(probs)
    log_probs[log_probs == -np.inf] = 0
    log_probs_sum = np.nansum(log_probs, axis=1)[:, np.newaxis]
    more_than_90 = np.sum(probs > .9, axis=1)[:, np.newaxis]
    return np.hstack((arc_length, probs_sum, mean_y, probs_mean,
                      more_than_90))[0]

  def _arm_thigh_calf_calibration_featurize(self):
    X = np.array(self.to_list())[np.newaxis]
    probs = X[:, 4::5]
    arc_length = np.sum(probs > 0, axis=1)[:, np.newaxis]
    probs_sum = np.sum(probs, axis=1)[:, np.newaxis]
    probs_mean = np.where(arc_length > 0, probs_sum / arc_length, 0)
    return np.hstack((arc_length, probs_sum, probs_mean))

  def calibration_featurize(self, region):
    if region == 'arm':
      return self._arm_thigh_calf_calibration_featurize()
    if region == 'torso':
      return self._torso_calibration_featurize()
    if region == 'thigh':
      return self._arm_thigh_calf_calibration_featurize()
    if region == 'calf':
      return self._arm_thigh_calf_calibration_featurize()

  def zone_featurize(self, region):
    if region == 'torso':
      return self._torso_zone_featurize()
    if region == 'thigh':
      return self._thigh_zone_featurize()

  def _torso_zone_featurize(self):
    X = np.array(self.to_list())[np.newaxis]
    probs = X[:, 4::5]
    active_slcs = probs != 0
    y_mins, y_maxes = X[:, 1::5], X[:, 3::5]
    mean_y_min = np.average(y_mins, axis=1, weights=y_mins != 0)
    mean_y_max = np.average(y_maxes, axis=1, weights=y_maxes != 0)
    mean_y = ((mean_y_min + mean_y_max) / 2)[:, np.newaxis]
    return np.hstack((active_slcs, mean_y))[0]

  def _thigh_zone_featurize(self):
    X = np.array(self.to_list())[np.newaxis]
    probs = X[:, 4::5]
    active_slcs = probs != 0
    x_mins = X[:, 0::5]
    y_mins = X[:, 1::5]
    x_maxes = X[:, 2::5]
    y_maxes = X[:, 3::5]
    mean_x = (x_maxes + x_mins)/2
    mean_y_min = np.average(y_mins, axis=1, weights=y_mins != 0)
    mean_y_max = np.average(y_maxes, axis=1, weights=y_maxes != 0)
    mean_y = ((mean_y_min + mean_y_max) / 2)[:, np.newaxis]
    return np.hstack((active_slcs, mean_y, mean_x))[0]


def equal_set(arcs_a, arcs_b):
  if len(arcs_a) != len(arcs_b):
    return False
  remaining_b = np.ones(len(arcs_b)).astype(np.bool)
  for arc in arcs_a:
    for i in range(len(arcs_b)):
      if remaining_b[i] and arc == arcs_b[i]:
        remaining_b[i] = False
        break
  return not np.any(remaining_b)


def _build_torso_thigh_arcs(logits,
                            bboxes,
                            max_gap=2,
                            y_iou_thresh=0.45,
                            score_threshold=0.3):
  assert bboxes.shape[0] == logits.shape[0] == 16
  nms_results = [
    non_max_suppression(
      bboxes[i],
      logits[i],
      min_clique_size=5,
      score_threshold=score_threshold,
      iou_threshold=0.4) for i in range(bboxes.shape[0])
  ]
  greedy_arcs = [
    _build_greedy_arcs(i, nms_results) for i in range(bboxes.shape[0])
  ]
  num_arcs_per_slc = [len(a) for a in greedy_arcs]
  min_arcs_inds = np.where(num_arcs_per_slc == np.min(num_arcs_per_slc))[0]
  arcs = greedy_arcs[min_arcs_inds[0]]
  split_arcs = _split_arcs(arcs, y_iou_thresh)
  merged_arcs = _merge_arcs(split_arcs, max_gap, y_iou_thresh)
  sorted_arcs = sorted(merged_arcs, key=lambda x: -x.total_prob())
  return sorted_arcs


def _merge_arcs(arcs, max_gap, y_iou_thresh, can_be_merged=None):
  if can_be_merged is None:
    can_be_merged = np.tril(
      np.ones((len(arcs), len(arcs))) - np.eye(len(arcs)))
  for i, j in np.array(list(zip(*np.where(can_be_merged)))):
    if not arcs[i].can_be_merged_with(arcs[j], max_gap, y_iou_thresh):
      can_be_merged[i, j] = 0
  if not np.any(can_be_merged):
    return arcs
  merge_inds = np.where(can_be_merged)
  i, j = merge_inds[0][0], merge_inds[1][0]
  arcs[i] += arcs[j]
  del arcs[j]
  can_be_merged[:, i] = 1
  can_be_merged[i, i] = 0
  can_be_merged = np.delete(can_be_merged, j, axis=0)
  can_be_merged = np.delete(can_be_merged, j, axis=1)
  return _merge_arcs(arcs, max_gap, y_iou_thresh, can_be_merged)


def _build_greedy_arcs(start_slc, nms_results):
  num_slcs, arcs = len(nms_results), []
  for slc in np.arange(start_slc, start_slc + num_slcs) % num_slcs:
    if len(nms_results[slc]) == 0:
      continue
    prev_boxes, prev_arc_inds = _arc_boxes_at_slc(arcs, (slc - 1) % num_slcs)
    curr_slc_boxes = np.array([r[0] for r in nms_results[slc]])
    if arcs == [] or prev_boxes.size == 0:
      for nms_result in nms_results[slc]:
        new_arc = Arc()
        new_arc[slc] = nms_result
        arcs.append(new_arc)
      continue
    box_distances = _interslc_box_distances(curr_slc_boxes, prev_boxes)
    remaining_curr_boxes = np.ones(curr_slc_boxes.shape[0]).astype(np.bool)
    remaining_prev_boxes = np.ones(prev_boxes.shape[0]).astype(np.bool)
    while np.any(remaining_curr_boxes) and np.any(remaining_prev_boxes):
      box_distances_remaining = box_distances[remaining_curr_boxes, :]
      box_distances_remaining = box_distances_remaining[:,
                                                        remaining_prev_boxes]
      min_curr, min_prev = np.unravel_index(
        np.argmin(box_distances_remaining), box_distances_remaining.shape)
      min_curr = np.arange(
        remaining_curr_boxes.shape[0])[remaining_curr_boxes][min_curr]
      min_prev = np.arange(
        remaining_prev_boxes.shape[0])[remaining_prev_boxes][min_prev]

      arcs[prev_arc_inds[min_prev]][slc] = nms_results[slc][min_curr]

      remaining_curr_boxes[min_curr] = False
      remaining_prev_boxes[min_prev] = False
    for i in np.where(remaining_curr_boxes)[0]:
      new_arc = Arc()
      new_arc[slc] = nms_results[slc][i]
      arcs.append(new_arc)

  return arcs


def _split_arcs(arcs, y_iou_thresh):
  split_arcs = []
  while len(arcs) > 0:
    arc = arcs.pop()
    inds = np.arange(arc.start_slc, arc.start_slc + len(arc)) % arc.num_slcs
    for i in range(len(inds) - 1):
      curr_box = arc[inds[i]][0]
      next_box = arc[inds[i + 1]][0]
      if np.min(
          y_iou(
            arc.boxes_through_index(inds[i]),
            np.array(arc[inds[i + 1]][0])[np.newaxis, :])
      ) < y_iou_thresh or next_box[0] > (curr_box[0] + curr_box[2]) / 2:
        split_arc = arc.split(inds[i + 1])
        arcs.append(split_arc)
        break
    split_arcs.append(arc)
  return split_arcs


def _split_arm_arcs(arcs, y_iou_thresh):
  split_arcs = []
  while len(arcs) > 0:
    arc = arcs.pop()
    inds = np.arange(arc.start_slc, arc.start_slc + len(arc)) % arc.num_slcs
    for i in range(len(inds) - 1):
      curr_box = arc[inds[i]][0]
      next_box = arc[inds[i + 1]][0]
      if np.min(
          y_iou(
            arc.boxes_through_index(inds[i]),
            np.array(arc[inds[i + 1]][0])[np.newaxis, :])) < y_iou_thresh:
        split_arc = arc.split(inds[i + 1])
        arcs.append(split_arc)
        break
    split_arcs.append(arc)
  return split_arcs


def _arc_boxes_at_slc(arcs, slc):
  boxes, arc_inds = [], []
  for i in range(len(arcs)):
    if arcs[i][slc]:
      boxes.append(arcs[i][slc][0])
      arc_inds.append(i)
  return np.array(boxes), np.array(arc_inds)


def _interslc_box_distances(boxes_a, boxes_b):
  # iou = _iou(boxes_a, boxes_b)
  iou = y_iou(boxes_a, boxes_b)
  return -iou + (iou == 0) * _mean_corner_delta(boxes_a, boxes_b)


def _mean_corner_delta(boxes_a, boxes_b):
  x_min_a, y_min_a, x_max_a, y_max_a = np.split(
    boxes_a, indices_or_sections=4, axis=1)
  x_min_b, y_min_b, x_max_b, y_max_b = np.split(
    boxes_b, indices_or_sections=4, axis=1)
  x_min_delta = x_min_a - np.transpose(x_min_b)
  y_min_delta = y_min_a - np.transpose(y_min_b)
  x_max_delta = x_max_a - np.transpose(x_max_b)
  y_max_delta = y_max_a - np.transpose(y_max_b)
  return (np.sqrt(x_min_delta**2 + y_min_delta**2) +
          np.sqrt(x_max_delta**2 + y_max_delta**2)) / 2


def y_iou(boxes_a, boxes_b):
  boxes_a = boxes_a.astype(np.float32)
  boxes_b = boxes_b.astype(np.float32)

  intersections = _y_intersection(boxes_a, boxes_b)
  y_lengths_a = _y_length(boxes_a)
  y_lengths_b = _y_length(boxes_b)
  unions = (
    y_lengths_a[:, np.newaxis] + y_lengths_b[np.newaxis, :] - intersections)
  return np.where(intersections == 0,
                  np.zeros_like(intersections), intersections / unions)


def _y_intersection(boxes_a, boxes_b):
  _, y_min_a, _, y_max_a = np.split(boxes_a, indices_or_sections=4, axis=1)
  _, y_min_b, _, y_max_b = np.split(boxes_b, indices_or_sections=4, axis=1)

  all_pairs_min_ymax = np.minimum(y_max_a, np.transpose(y_max_b))
  all_pairs_max_ymin = np.maximum(y_min_a, np.transpose(y_min_b))
  intersect_heights = np.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
  return intersect_heights


def _y_length(boxes):
  _, y_min, _, y_max = np.split(boxes, indices_or_sections=4, axis=1)
  return np.reshape((y_max - y_min), (-1, ))


def gaps_between(start_slc_a, end_slc_a, start_slc_b, end_slc_b, num_slcs):
  return ((start_slc_a - end_slc_b) % num_slcs - 1,
          (start_slc_b - end_slc_a) % num_slcs - 1)


def normalized_y(cropped_bbox, box):
  y_pad = pad_dim((cropped_bbox[3] - cropped_bbox[1],
                   cropped_bbox[2] - cropped_bbox[0]))[0]
  normalized_bbox = (box - y_pad) / (cropped_bbox[3] - cropped_bbox[1])
  return normalized_bbox[1], normalized_bbox[3]


def rescale_cropped_box(cropped_bbox):
  pad = pad_dim((cropped_bbox[3] - cropped_bbox[1],
                 cropped_bbox[2] - cropped_bbox[0]))
  original_size = 2 * pad[0] + cropped_bbox[3] - cropped_bbox[1]
  rescaled_pad = pad[0] * 600. / original_size, pad[1] * 600. / original_size
  return [
    rescaled_pad[1], rescaled_pad[0], 600 - rescaled_pad[1],
    600 - rescaled_pad[0]
  ]


def cluster_calfs(image):
  copied_image = np.copy(image)
  copied_image -= np.mean(copied_image)
  model = KMeans(2)
  X = np.where(copied_image > 0)[1]
  X = X[:, np.newaxis]
  model.fit(X)
  centers = model.cluster_centers_
  return np.array([min(centers)[0], max(centers)[0]])


def get_calf_side_and_height(_id, slc, box):
  image, cropped_bbox = crop_and_pad(
    _id, 'calf', 'aps', slc, keypoint_pad=None)
  image = resize(image, (600, 600), mode='constant')
  centers = cluster_calfs(image)
  box_x_mean = (box[0] + box[2]) / 2
  y = np.mean(np.array(normalized_y(rescale_cropped_box(cropped_bbox), box)))
  if slc == 3:
    if box[0] < centers[0]:
      return 'R', y
    if box[2] > centers[1]:
      return 'L', y
    return None, y
  if slc == 5:
    if box[0] < centers[0]:
      return 'L', y
    if box[2] > centers[1]:
      return 'R', y
    return None, y
  if slc == 11:
    if box[0] < centers[0]:
      return 'L', y
    if box[2] > centers[1]:
      return 'R', y
    return None, y
  if slc == 13:
    if box[0] < centers[0]:
      return 'R', y
    if box[2] > centers[1]:
      return 'L', y
    return None, y
  if slc in [14, 15, 0, 1, 2]:
    if np.argmin(np.abs(centers - box_x_mean)) == 0:
      side = 'R'
    else:
      side = 'L'
  if slc in [6, 7, 8, 9, 10]:
    if np.argmin(np.abs(centers - box_x_mean)) == 0:
      side = 'L'
    else:
      side = 'R'
  if slc in [4, 12]:
    side = None
  return side, y


def get_closer_side(_id, slc, box):
  assert slc in [3, 5, 11, 13]
  image, cropped_bbox = crop_and_pad(
    _id, 'calf', 'aps', slc, keypoint_pad=None)
  image = resize(image, (600, 600), mode='constant')
  centers = cluster_calfs(image)
  box_x_mean = (box[0] + box[2]) / 2
  if slc in [3, 13]:
    if np.argmin(np.abs(centers - box_x_mean)) == 0:
      return 'R'
    else:
      return 'L'
  if slc in [5, 11]:
    if np.argmin(np.abs(centers - box_x_mean)) == 0:
      return 'L'
    else:
      return 'R'


def _build_calf_arcs(logits, bboxes, _id):
  arcs = Arc(), Arc(), Arc(), Arc()
  arc_13, arc_14, arc_15, arc_16 = arcs

  def _place_if_better(suppressed_result, arc, slc):
    if arc[slc] and suppressed_result[1] > arc[slc][1]:
      del arc[slc]
      arc[slc] = suppressed_result
    elif not arc[slc]:
      arc[slc] = suppressed_result

  def _place_suppressed_result(suppressed_result, side, height, arcs):
    assert side
    arc_13, arc_14, arc_15, arc_16 = arcs
    if side == 'R' and height <= 0.5:
      _place_if_better(suppressed_result, arc_13, slc)
    elif side == 'R':
      _place_if_better(suppressed_result, arc_15, slc)
    elif height <= 0.5:
      _place_if_better(suppressed_result, arc_14, slc)
    else:
      _place_if_better(suppressed_result, arc_16, slc)

  nms_results = [
    non_max_suppression(
      bboxes[i],
      logits[i],
      min_clique_size=5,
      score_threshold=0.1,
      iou_threshold=0.4,
      ios_threshold=0.2) for i in range(bboxes.shape[0])
  ]
  for slc in [14, 15, 0, 1, 2, 6, 7, 8, 9, 10]:
    nms_result = nms_results[slc]
    for i in range(len(nms_result)):
      side, height = get_calf_side_and_height(_id, slc, nms_result[i][0])
      _place_suppressed_result(nms_result[i], side, height, arcs)
  unplaced_results = []
  for slc in [3, 5, 11, 13]:
    nms_result = nms_results[slc]
    for i in range(len(nms_result)):
      side, height = get_calf_side_and_height(_id, slc, nms_result[i][0])
      if side:
        _place_suppressed_result(nms_result[i], side, height, arcs)
      else:
        unplaced_results.append((slc, nms_result[i], height))
  for slc in [4]:
    nms_result = nms_results[slc]
    for i in range(len(nms_result)):
      _, height = get_calf_side_and_height(_id, slc, nms_result[i][0])
      if height <= 0.5:
        if len(arc_14) == 0 and (arc_13[3] or arc_13[5]):
          _place_suppressed_result(nms_result[i], 'R', height, arcs)
        else:
          _place_suppressed_result(nms_result[i], 'L', height, arcs)
      else:
        if len(arc_16) == 0 and (arc_15[3] or arc_15[5]):
          _place_suppressed_result(nms_result[i], 'R', height, arcs)
        else:
          _place_suppressed_result(nms_result[i], 'L', height, arcs)
  for slc in [12]:
    nms_result = nms_results[slc]
    for i in range(len(nms_result)):
      _, height = get_calf_side_and_height(_id, slc, nms_result[i][0])
      if height <= 0.5:
        if len(arc_13) == 0 and (arc_14[3] or arc_14[5]):
          _place_suppressed_result(nms_result[i], 'L', height, arcs)
        else:
          _place_suppressed_result(nms_result[i], 'R', height, arcs)
      else:
        if len(arc_15) == 0 and (arc_16[3] or arc_16[5]):
          _place_suppressed_result(nms_result[i], 'L', height, arcs)
        else:
          _place_suppressed_result(nms_result[i], 'R', height, arcs)
  for slc, result, height in unplaced_results:
    if height <= 0.5:
      if len(arc_13) < 2 and len(arc_14) >= 2:
        _place_if_better(result, arc_14, slc)
      elif len(arc_13) >= 2 and len(arc_14) < 2:
        _place_if_better(result, arc_13, slc)
      elif get_closer_side(_id, slc, result[0]) == 'R':
        _place_if_better(result, arc_13, slc)
      else:
        _place_if_better(result, arc_14, slc)
    else:
      if len(arc_15) < 2 and len(arc_16) >= 2:
        _place_if_better(result, arc_16, slc)
      elif len(arc_15) >= 2 and len(arc_16) < 2:
        _place_if_better(result, arc_15, slc)
      elif get_closer_side(_id, slc, result[0]) == 'R':
        _place_if_better(result, arc_15, slc)
      else:
        _place_if_better(result, arc_16, slc)
  return arcs


def _build_arm_arcs(logits, bboxes, _id):
  arcs = Arc(), Arc(), Arc(), Arc()
  arc_1, arc_2, arc_3, arc_4 = arcs

  def _place_if_better(suppressed_result, arc, slc):
    if arc[slc] and suppressed_result[1] > arc[slc][1]:
      del arc[slc]
      arc[slc] = suppressed_result
    elif not arc[slc]:
      arc[slc] = suppressed_result

  nms_results = [
    non_max_suppression(
      bboxes[i],
      logits[i],
      min_clique_size=5,
      score_threshold=0.1,
      iou_threshold=0.4,
      ios_threshold=0.2) for i in range(bboxes.shape[0])
  ]
  for slc in range(16):
    nms_result = nms_results[slc]
    for i in range(len(nms_result)):
      box = nms_result[i][0].astype(np.int32) // 3
      zone = localization_inference(_id, 'arm', box, slc)
      if zone == 1:
        _place_if_better(nms_result[i], arc_1, slc)
      if zone == 2:
        _place_if_better(nms_result[i], arc_2, slc)
      if zone == 3:
        _place_if_better(nms_result[i], arc_3, slc)
      if zone == 4:
        _place_if_better(nms_result[i], arc_4, slc)
  return arcs


def _build_torso_arcs(logits, bboxes):
  return _build_torso_thigh_arcs(logits, bboxes)


def _build_thigh_arcs(logits, bboxes):
  return _build_torso_thigh_arcs(logits, bboxes, score_threshold=0.1)


def build_arcs(region, logits, bboxes, _id=None):
  if region == 'arm':
    return _build_arm_arcs(logits, bboxes, _id)
  if region == 'torso':
    return _build_torso_arcs(logits, bboxes)
  if region == 'thigh':
    return _build_thigh_arcs(logits, bboxes)
  if region == 'calf':
    return _build_calf_arcs(logits, bboxes, _id)