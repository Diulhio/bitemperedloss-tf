# -*- coding: utf-8 -*-

import tensorflow as tf

def log_t(u, t):
  """Compute log_t for `u`."""
  if t == 1.0:
    return tf.math.log(u)
  else:
    return (u**(1.0 - t) - 1.0) / (1.0 - t)

def exp_t(u, t):
  """Compute exp_t for `u`."""
  if t == 1.0:
    return tf.math.exp(u)
  else:
    return tf.math.maximum(0.0, 1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))

def compute_normalization_fixed_point(y_pred, t, num_iters=5):
  """Returns the normalization value for each example (t > 1.0).
    Args:
    y_pred: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (> 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as y_pred with the last dimension being 1.
  """
  mu = tf.math.reduce_max(y_pred, -1, keepdims=True)
  normalized_y_pred_step_0 = y_pred - mu
  normalized_y_pred = normalized_y_pred_step_0
  i = 0
  while i < num_iters:
    i += 1
    logt_partition = tf.math.reduce_sum(exp_t(normalized_y_pred, t),-1, keepdims=True)
    normalized_y_pred = normalized_y_pred_step_0 * (logt_partition ** (1.0 - t))
  
  logt_partition = tf.math.reduce_sum(exp_t(normalized_y_pred, t), -1, keepdims=True)
  return -log_t(1.0 / logt_partition, t) + mu

def compute_normalization(y_pred, t, num_iters=5):
  """Returns the normalization value for each example.
    Args:
    y_pred: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
  """
  if t < 1.0:
    return None # not implemented as these values do not occur in the authors experiments...
  else:
    return compute_normalization_fixed_point(y_pred, t, num_iters)

def tempered_softmax(y_pred, t, num_iters=5):
  """Tempered softmax function.
    Args:
    y_pred: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature tensor > 0.0.
    num_iters: Number of iterations to run the method.
    Returns:
    A probabilities tensor.
  """
  if t == 1.0:
    normalization_constants = tf.math.log(tf.math.reduce_sum(tf.math.exp(y_pred), -1, keepdims=True))
  else:
    normalization_constants = compute_normalization(y_pred, t, num_iters)

  return exp_t(y_pred - normalization_constants, t)

def bi_tempered_logistic_loss(y_pred, y_true, t1, t2, num_iters=5, label_smoothing=0.0):
  """Bi-Tempered Logistic Loss with custom gradient.
    Args:
    y_pred: A multi-dimensional tensor with last dimension `num_classes`.
    y_true: A tensor with shape and dtype as y_pred.
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    num_iters: Number of iterations to run the method.
    Returns:
    A loss tensor.
  """
  y_pred = tf.cast(y_pred, tf.float32)
  y_true = tf.cast(y_true, tf.float32)

  if label_smoothing > 0.0:
    num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
    y_true = (1 - num_classes /(num_classes - 1) * label_smoothing) * y_true + label_smoothing / (num_classes - 1)

  probabilities = tempered_softmax(y_pred, t2, num_iters)

  temp1 = (log_t(y_true + 1e-10, t1) - log_t(probabilities, t1)) * y_true
  temp2 = (1 / (2 - t1)) * (tf.math.pow(y_true, 2 - t1) - tf.math.pow(probabilities, 2 - t1))
  loss_values = temp1 - temp2

  return tf.math.reduce_sum(loss_values, -1)

class BiTemperedLogisticLoss(tf.keras.losses.Loss):
  def __init__(self, t1, t2, n_iter=5, label_smoothing=0.0):
    super(BiTemperedLogisticLoss, self).__init__()
    self.t1 = t1
    self.t2 = t2
    self.n_iter = n_iter
    self.label_smoothing = label_smoothing

  def call(self, y_true, y_pred):
    return bi_tempered_logistic_loss(y_pred, y_true, self.t1, self.t2, self.n_iter, self.label_smoothing)