import numpy as np


def non_max_suppression(boxes,
                        scores,
                        score_threshold=0.3,
                        iou_threshold=0.5,
                        ios_threshold=0.5,
                        ioo_threshold=0.5,
                        min_clique_size=5):
  keep_inds = np.where(scores >= score_threshold)
  boxes, scores = boxes[keep_inds], scores[keep_inds]
  sorted_scores_inds = np.argsort(scores)[::-1]
  boxes, scores = boxes[sorted_scores_inds, :], scores[sorted_scores_inds]
  iou_grid = iou(boxes)
  ios_grid = ios(boxes)
  ioo_grid = np.transpose(ios_grid)

  suppressed_boxes = []

  remain_row_indicators = np.ones(iou_grid.shape[0])
  while np.sum(remain_row_indicators) > 0:
    r = np.where(remain_row_indicators == 1)[0][0]
    iou_row, ios_row, ioo_row = iou_grid[r], ios_grid[r], ioo_grid[r]
    if np.sum(iou_row > iou_threshold) > min_clique_size:
      suppressed_boxes.append((boxes[r], scores[r]))
      remain_row_indicators[np.where(iou_row > iou_threshold)[0]] = 0
      remain_row_indicators[np.where(ios_row > ios_threshold)[0]] = 0
      remain_row_indicators[np.where(ioo_row > ioo_threshold)[0]] = 0
    remain_row_indicators[r] = 0
  return suppressed_boxes


def iou(boxes):
  boxes = boxes.astype(np.float32)
  intersections = _intersection(boxes)
  areas = _area(boxes)[:, np.newaxis]

  unions = ((areas + np.transpose(areas)) - intersections)
  return np.where(
    np.equal(intersections, 0.0),
    np.zeros_like(intersections), np.true_divide(intersections, unions))


def ios(boxes):
  """Intersection over self"""
  boxes = boxes.astype(np.float32)
  return _intersection(boxes) / _area(boxes)


def _intersection(boxes):
  y_min, x_min, y_max, x_max = np.split(boxes, indices_or_sections=4, axis=1)
  all_pairs_min_ymax = np.minimum(y_max, np.transpose(y_max))
  all_pairs_max_ymin = np.maximum(y_min, np.transpose(y_min))
  intersect_heights = np.maximum(0.0, all_pairs_min_ymax - all_pairs_max_ymin)
  all_pairs_min_xmax = np.minimum(x_max, np.transpose(x_max))
  all_pairs_max_xmin = np.maximum(x_min, np.transpose(x_min))
  intersect_widths = np.maximum(0.0, all_pairs_min_xmax - all_pairs_max_xmin)
  return intersect_heights * intersect_widths


def _area(boxes):
  x_min, y_min, x_max, y_max = np.split(boxes, indices_or_sections=4, axis=1)
  return np.reshape((y_max - y_min) * (x_max - x_min), (-1, ))