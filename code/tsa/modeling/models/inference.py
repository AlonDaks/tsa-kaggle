from datascience import *
import numpy as np
import tensorflow as tf
from tsa.modeling.models.threat_detect_inference import threat_detect_inference
from tsa.modeling.models.keypoint_crop import crop_and_pad
from tsa.utils.arc import build_arcs
from tsa.modeling.models.calibrate import TorsoCalibrationModel, CalfCalibrationModel, ArmCalibrationModel, ThighCalibrationModel
import tsa.utils.data_path as dp
from skimage.transform import resize

MIN_PREDICTION = 0.004


def _calf_inference(_id):
  logits, bboxes = threat_detect_inference(_id, 'calf')
  arcs = build_arcs('calf', logits, bboxes, _id)
  prediction_strings = []
  for i in range(len(arcs)):
    arc = arcs[i]
    zone = i + 13
    predict_proba = CalfCalibrationModel().calibrate(
      arc.calibration_featurize('calf'))
    prediction_strings.append(
      '{0}_Zone{1},{2}'.format(_id, zone, predict_proba))
  for s in prediction_strings:
    print(s)


def _arm_inference(_id):
  logits, bboxes = threat_detect_inference(_id, 'arm')
  arcs = build_arcs('arm', logits, bboxes, _id)
  prediction_strings = []
  for i in range(len(arcs)):
    arc = arcs[i]
    zone = i + 1
    predict_proba = ArmCalibrationModel().calibrate(
      arc.calibration_featurize('arm'))
    prediction_strings.append(
      '{0}_Zone{1},{2}'.format(_id, zone, predict_proba))
  for s in prediction_strings:
    print(s)


def _torso_inference(_id):
  logits, bboxes = threat_detect_inference(_id, 'torso')
  arcs = build_arcs('torso', logits, bboxes)
  zone_predictions = {i: [] for i in [5, 6, 7, 17]}
  for i in range(len(arcs)):
    arc = arcs[i]
    predict_proba = TorsoCalibrationModel().calibrate(
      arc.calibration_featurize('torso'))
    zone = TorsoCalibrationModel().zone(arc.zone_featurize('torso'))[0]
    zone_predictions[zone].append(predict_proba)
  prediction_strings = []
  for zone in [5, 6, 7, 17]:
    preds = np.array(sorted(zone_predictions[zone])[::-1])
    if len(preds) == 0:
      predict_proba = MIN_PREDICTION
    else:
      predict_proba = preds[0] + (1 - preds[0]) * min(np.sum(preds[1:]), 1)
    prediction_strings.append(
      '{0}_Zone{1},{2}'.format(_id, zone, predict_proba))
  for s in prediction_strings:
    print(s)


def _thigh_inference(_id):
  logits, bboxes = threat_detect_inference(_id, 'thigh')
  arcs = build_arcs('thigh', logits, bboxes)
  zone_predictions = {i: [] for i in [8, 9, 10, 11, 12]}
  for i in range(len(arcs)):
    arc = arcs[i]
    predict_proba = ThighCalibrationModel().calibrate(
      arc.calibration_featurize('thigh'))
    zone = ThighCalibrationModel().zone(arc.zone_featurize('thigh'))[0]
    zone_predictions[zone].append(predict_proba)
  prediction_strings = []
  for zone in [8, 9, 10, 11, 12]:
    preds = np.array(sorted(zone_predictions[zone])[::-1])
    if len(preds) == 0:
      predict_proba = MIN_PREDICTION
    else:
      predict_proba = preds[0] + (1 - preds[0]) * min(np.sum(preds[1:]), 1)
    prediction_strings.append(
      '{0}_Zone{1},{2}'.format(_id, zone, predict_proba))
  for s in prediction_strings:
    print(s)


def _inference(data_set, region):
  if region == 'arm':
    region_inference_func, zones = _arm_inference, [1, 2, 3, 4]
  if region == 'torso':
    region_inference_func, zones = _torso_inference, [5, 6, 7, 17]
  if region == 'thigh':
    region_inference_func, zones = _thigh_inference, [8, 9, 10, 11, 12]
  if region == 'calf':
    region_inference_func, zones = _calf_inference, [13, 14, 15, 16]

  ids = Table().read_table(dp.ids_by_data_set(data_set)).column('Id')
  for _id in ids:
    try:
      region_inference_func(_id)
    except:
      for zone in zones:
        print('{0}_Zone{1},{2}'.format(_id, zone, 0.1))


def main(argv=None):  # pylint: disable=unused-argument
  assert FLAGS.region
  _inference(FLAGS.data_set, FLAGS.region)


if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_string('data_set', None,
                             """data_set to run inference on""")
  tf.app.flags.DEFINE_string('region', None, """arm, torso, thigh, or calf""")

  tf.app.run()