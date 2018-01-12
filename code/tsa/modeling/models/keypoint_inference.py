import tensorflow as tf

from tsa.modeling.models.keypoint_params import *
from tsa.modeling.models.anchor import *
from tsa.modeling.models.keypoint_detect_net import *
import tsa.utils.data_path as dp
from tsa.utils.non_max_suppression import non_max_suppression
from tsa.utils.merge_bbox_xml import parse_bbox_xml, write_xml
import numpy as np
from PIL import Image
from datascience import *


def _inference(data_set, keypoint):
  tf.reset_default_graph()

  ids = Table().read_table(dp.ids_by_data_set(data_set)).column('Id')
  base_png_path = dp.LARGE_DATA_BIN + '/data/raw/'
  if keypoint.file_format == 'a3daps':
    base_png_path += 'a3daps_png/'
  else:
    base_png_path += 'aps_png/'
  base_xml_output_path = dp.REPO_HOME_PATH + '/data/bbox/keypoint_inference/' + keypoint.name

  images_tensor = tf.placeholder(tf.float32, (None, 660, 512, 1))
  model = KeypointDetectModel(
    batch_size=1, image_dim=(660, 512), anchor_kwargs=keypoint.anchor_kwargs)
  logits_tensor, bboxes_tensor = model.inference(images_tensor, test=True)
  logits_tensor = tf.sigmoid(logits_tensor)
  bboxes_tensor = target_to_box(bboxes_tensor, model.anchors)

  saver = tf.train.Saver(tf.global_variables())

  with tf.Session() as sess:
    saver.restore(sess,
                  tf.train.get_checkpoint_state(
                    dp.train_dir('{0}_keypoint'.format(keypoint.name)))
                  .model_checkpoint_path)
    for _id in ids:
      image = np.array(
        Image.open('{0}/{1}/{1}_{2}.png'.format(base_png_path, keypoint.slices[
          0], _id))).astype(np.float32)
      image -= np.mean(image)
      logits, bboxes = sess.run(
        [logits_tensor, bboxes_tensor],
        feed_dict={images_tensor: image[np.newaxis, :, :, np.newaxis]})
      nms_result = non_max_suppression(
        bboxes,
        logits,
        min_clique_size=1,
        score_threshold=0.05,
        iou_threshold=0.4)
      if len(nms_result) > 0:
        bbox = [int(round(i)) for i in nms_result[0][0]]
      elif keypoint.name == 'butt':
        bbox = [180, 359, 340, 438]
      elif keypoint.name == 'face':
        bbox = [204, 137, 302, 220]
      else:
        bbox = 4 * [0]
      write_xml(None, None, None,
                [bbox])(base_xml_output_path + '/{}.xml'.format(_id))


def main(argv=None):  # pylint: disable=unused-argument
  if FLAGS.keypoint == 'face':
    keypoint = FACE
  if FLAGS.keypoint == 'butt':
    keypoint = BUTT
  _inference(FLAGS.data_set, keypoint)


if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS

  tf.app.flags.DEFINE_string('data_set', None, 'data_set to run inference on')
  tf.app.flags.DEFINE_string('keypoint', None, 'face or butt')

  tf.app.run()
