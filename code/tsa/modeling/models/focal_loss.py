import tensorflow as tf
from tsa.modeling.models.anchor import box_to_target

EPSILON = 1e-8

def loss_func(anchor_gt_box, bboxes, gt_cls_labels, logits, anchors, lam=1.0):
  gt_cls_labels = tf.cast(gt_cls_labels, tf.float32)
  logits = tf.cast(logits, tf.float32)
  cls_mask = tf.where(
    tf.not_equal(gt_cls_labels, -1),
    tf.ones_like(gt_cls_labels), tf.zeros_like(gt_cls_labels))
  masked_gt_cls_labels = cls_mask * gt_cls_labels

  def focal_loss(gt_cls_labels, logits, alpha=0.25, gamma=2.0, scope=None):
    with tf.name_scope(scope, 'focal_loss') as scope:
      losses = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=gt_cls_labels, logits=logits)
      preds = tf.sigmoid(logits)
      p_t = tf.where(tf.equal(gt_cls_labels, 1), preds, 1.0 - preds)
      alpha_t = alpha * gt_cls_labels + (1 - alpha) * (1 - gt_cls_labels)
      modulated_losses = alpha_t * (1 - p_t)**gamma * losses
      masked_modulated_losses = cls_mask * modulated_losses
      focal_losses = (tf.reduce_sum(masked_modulated_losses, axis=1) /
                      (tf.reduce_sum(masked_gt_cls_labels, axis=1) + 1))
      return focal_losses

  focal_losses = focal_loss(gt_cls_labels, logits)

  def smooth_l1_loss(anchor_gt_box, bboxes, gt_cls_labels, anchors,
                     scope=None):
    with tf.name_scope(scope, 'smooth_l1_loss') as scope:
      t_anchor_gt_box = box_to_target(anchor_gt_box, anchors)
      t_anchor_gt_box = tf.check_numerics(t_anchor_gt_box, 't_anchor_gt_box',
                                          't_anchor_gt_box')
      losses = tf.reduce_sum(
        tf.losses.huber_loss(
          t_anchor_gt_box, bboxes, reduction=tf.losses.Reduction.NONE),
        axis=2)

      losses = tf.check_numerics(losses, 'losses')
      reg_mask = tf.where(
        tf.equal(gt_cls_labels, 1),
        tf.ones_like(gt_cls_labels), tf.zeros_like(gt_cls_labels))
      masked_losses = reg_mask * losses
      num_positive_anchors = tf.reduce_sum(masked_gt_cls_labels, axis=1)

      regression_losses = tf.reduce_sum(
        masked_losses, axis=1) / (num_positive_anchors + EPSILON)

      regression_losses = tf.check_numerics(regression_losses,
                                            'regression_losses')
      return regression_losses

  regression_losses = smooth_l1_loss(anchor_gt_box, bboxes, gt_cls_labels,
                                     anchors)

  loss = tf.reduce_mean(focal_losses + lam * regression_losses)
  loss = tf.check_numerics(loss, 'FINAL_LOSS', 'FINAL_LOSS')
  return loss