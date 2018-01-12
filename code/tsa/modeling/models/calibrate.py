import tsa.utils.data_path as dp
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tsa.modeling.models.localization_inference import REGION_ZONES, localization_inference


class TorsoCalibrationModel:
  class __TorsoCalibrationModel:
    def __init__(self, calibration_model, zone_model):
      self.calibration_model = calibration_model
      self.zone_model = zone_model

    def calibrate(self, example_x):
      return self.calibration_model.predict_proba(
        example_x.reshape(1, -1))[0, 1]

    def zone(self, example_x):
      return self.zone_model.predict(example_x.reshape(1, -1))

  instance = None

  def __calibration_featurize(X):
    probs = X[:, 4::5]
    arc_length = np.sum(probs > 0, axis=1)[:, np.newaxis]
    probs_sum = np.sum(probs, axis=1)[:, np.newaxis]
    y_mins, y_maxes = X[:, 1::5], X[:, 3::5]
    mean_y_min = np.average(y_mins, axis=1, weights=y_mins != 0)
    mean_y_max = np.average(y_maxes, axis=1, weights=y_maxes != 0)
    mean_y = ((mean_y_min + mean_y_max) / 2)[:, np.newaxis]
    max_prob = np.max(probs, axis=1)[:, np.newaxis]
    probs_mean = probs_sum / arc_length
    more_than_90 = np.sum(probs > .9, axis=1)[:, np.newaxis]
    return np.hstack((arc_length, probs_sum, mean_y, probs_mean, more_than_90))

  def __localization_featurize(X):
    probs = X[:, 4::5]
    active_slcs = probs != 0
    y_mins, y_maxes = X[:, 1::5], X[:, 3::5]
    mean_y_min = np.average(y_mins, axis=1, weights=y_mins != 0)
    mean_y_max = np.average(y_maxes, axis=1, weights=y_maxes != 0)
    mean_y = ((mean_y_min + mean_y_max) / 2)[:, np.newaxis]
    return np.hstack((active_slcs, mean_y))

  def __init__(self):
    if not TorsoCalibrationModel.instance:
      X_train_raw, y_train_label, y_train_zone, _ = load_calibration_data(
        'train_0', 'torso')
      X_validate_raw, y_validate_label, y_validate_zone, _ = load_calibration_data(
        'validate_0', 'torso')
      X_total_raw = np.vstack((X_train_raw, X_validate_raw))
      X_total = TorsoCalibrationModel.__calibration_featurize(X_total_raw)
      X_total_localization = TorsoCalibrationModel.__localization_featurize(
        X_total_raw)
      y_total_label = np.append(y_train_label, y_validate_label)
      y_total_localization = np.append(y_train_zone, y_validate_zone)

      X_total_localization = X_total_localization[y_total_localization != None]
      y_total_localization = y_total_localization[y_total_localization != None]
      y_total_localization = [int(y) for y in y_total_localization]

      np.random.seed(102)
      calibration_model = LogisticRegression(C=1, max_iter=1000)
      calibration_model.fit(X=X_total, y=y_total_label)

      localization_model = RandomForestClassifier(n_estimators=1000)
      localization_model.fit(X=X_total_localization, y=y_total_localization)

      TorsoCalibrationModel.instance = TorsoCalibrationModel.__TorsoCalibrationModel(
        calibration_model, localization_model)

  def __getattr__(self, name):
    return getattr(self.instance, name)


class CalfCalibrationModel:
  class __CalfCalibrationModel:
    def __init__(self, model):
      self.model = model

    def calibrate(self, example_x):
      return self.model.predict_proba(example_x)[0, 1]

  instance = None

  def __featurize(X):
    probs = X[:, 4::5]
    arc_length = np.sum(probs > 0, axis=1)[:, np.newaxis]
    probs_sum = np.sum(probs, axis=1)[:, np.newaxis]
    probs_mean = np.where(arc_length > 0, probs_sum / arc_length, 0)
    return np.hstack((arc_length, probs_sum, probs_mean))

  def __init__(self):
    if not CalfCalibrationModel.instance:
      X_train_raw, y_train_label, y_train_zone, ids_train = load_calibration_data(
        'train_0', 'calf')
      X_validate_raw, y_validate_label, y_validate_zone, ids_validate = load_calibration_data(
        'validate_0', 'calf')
      X_total_raw = np.vstack((X_train_raw, X_validate_raw))
      X_total = CalfCalibrationModel.__featurize(X_total_raw)
      y_total_label = np.append(y_train_label, y_validate_label)
      np.random.seed(101)
      model = LogisticRegression(C=1, max_iter=1000)
      model.fit(X=X_total, y=y_total_label)
      CalfCalibrationModel.instance = CalfCalibrationModel.__CalfCalibrationModel(
        model)

  def __getattr__(self, name):
    return getattr(self.instance, name)


class ArmCalibrationModel:
  class __ArmCalibrationModel:
    def __init__(self, model):
      self.model = model

    def calibrate(self, example_x):
      return self.model.predict_proba(example_x)[0, 1]

  instance = None

  def __featurize(X):
    probs = X[:, 4::5]
    arc_length = np.sum(probs > 0, axis=1)[:, np.newaxis]
    probs_sum = np.sum(probs, axis=1)[:, np.newaxis]
    probs_mean = np.where(arc_length > 0, probs_sum / arc_length, 0)
    return np.hstack((arc_length, probs_sum, probs_mean))

  def __init__(self):
    if not ArmCalibrationModel.instance:
      X_train_raw, y_train_label, y_train_zone, ids_train = load_calibration_data(
        'train_0', 'arm')
      X_validate_raw, y_validate_label, y_validate_zone, ids_validate = load_calibration_data(
        'validate_0', 'arm')
      X_total_raw = np.vstack((X_train_raw, X_validate_raw))
      X_total = ArmCalibrationModel.__featurize(X_total_raw)
      y_total_label = np.append(y_train_label, y_validate_label)
      np.random.seed(101)
      model = LogisticRegression(C=1, max_iter=1000)
      model.fit(X=X_total, y=y_total_label)
      ArmCalibrationModel.instance = ArmCalibrationModel.__ArmCalibrationModel(
        model)

  def __getattr__(self, name):
    return getattr(self.instance, name)


class ThighCalibrationModel:
  class __ThighCalibrationModel:
    def __init__(self, model, zone_model):
      self.model = model
      self.zone_model = zone_model

    def calibrate(self, example_x):
      return self.model.predict_proba(example_x)[0, 1]

    def zone(self, example_x):
      return self.zone_model.predict(example_x.reshape(1, -1))

  instance = None

  def __featurize(X):
    probs = X[:, 4::5]
    arc_length = np.sum(probs > 0, axis=1)[:, np.newaxis]
    probs_sum = np.sum(probs, axis=1)[:, np.newaxis]
    probs_mean = np.where(arc_length > 0, probs_sum / arc_length, 0)
    return np.hstack((arc_length, probs_sum, probs_mean))

  def __localization_featurize(X):
    probs = X[:, 4::5]
    active_slcs = probs != 0
    x_mins = X[:, 0::5]
    y_mins = X[:, 1::5]
    x_maxes = X[:, 2::5]
    y_maxes = X[:, 3::5]
    mean_x = (x_maxes + x_mins) / 2
    mean_y_min = np.average(y_mins, axis=1, weights=y_mins != 0)
    mean_y_max = np.average(y_maxes, axis=1, weights=y_maxes != 0)
    mean_y = ((mean_y_min + mean_y_max) / 2)[:, np.newaxis]
    return np.hstack((active_slcs, mean_y, mean_x))

  def __init__(self):
    if not ThighCalibrationModel.instance:
      X_train_raw, y_train_label, y_train_zone, ids_train = load_calibration_data(
        'train_0', 'thigh')
      X_validate_raw, y_validate_label, y_validate_zone, ids_validate = load_calibration_data(
        'validate_0', 'thigh')
      X_total_raw = np.vstack((X_train_raw, X_validate_raw))
      X_total = ThighCalibrationModel.__featurize(X_total_raw)
      X_total_localization = ThighCalibrationModel.__localization_featurize(
        X_total_raw)
      y_total_label = np.append(y_train_label, y_validate_label)
      y_total_localization = np.append(y_train_zone, y_validate_zone)

      X_total_localization = X_total_localization[y_total_localization != None]
      y_total_localization = y_total_localization[y_total_localization != None]
      y_total_localization = [int(y) for y in y_total_localization]

      np.random.seed(102)
      calibration_model = LogisticRegression(C=1, max_iter=1000)
      calibration_model.fit(X=X_total, y=y_total_label)

      localization_model = RandomForestClassifier(n_estimators=1000)
      localization_model.fit(X=X_total_localization, y=y_total_localization)

      ThighCalibrationModel.instance = ThighCalibrationModel.__ThighCalibrationModel(
        calibration_model, localization_model)

  def __getattr__(self, name):
    return getattr(self.instance, name)


def load_calibration_data(data_set, region):
  outfile = np.load(dp.REPO_HOME_PATH + '/data/calibration/{0}_{1}.npz'.format(
    data_set, region))
  return (outfile['arr_{}'.format(i)] for i in range(4))
