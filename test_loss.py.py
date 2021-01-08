import tensorflow as tf
from tf_bi_tempered_loss import BiTemperedLogisticLoss
import numpy as np


y_pred = tf.constant([-0.5,  0.1,  2.0], dtype=tf.float32)
y_true = tf.constant([0.2, 0.5, 0.3], dtype=tf.float32)
bi_temp = BiTemperedLogisticLoss(t1=1.0, t2=1.0)
print("Test1 -> Loss, t1=1.0, t2=1.0: ", bi_temp(y_true, y_pred).numpy())
np.testing.assert_almost_equal(bi_temp(y_true, y_pred).numpy(), 0.62870467)

bi_temp = BiTemperedLogisticLoss(t1=0.7, t2=1.3)
print("Test2 -> Loss, t1=0.7, t2=1.3: ", bi_temp(y_true, y_pred).numpy())
np.testing.assert_almost_equal(bi_temp(y_true, y_pred).numpy(), 0.2629559)

# Please note that now the loss function will return the mean loss!
y_pred = tf.constant([[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0]], dtype=tf.float32)
y_true = tf.constant([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1], [0.2, 0.8, 0.0]], dtype=tf.float32)
bi_temp = BiTemperedLogisticLoss(t1=0.5, t2=1.5)
print("Test3 -> Loss, t1=0.5, t2=1.5: ", bi_temp(y_true, y_pred).numpy())
np.testing.assert_almost_equal(bi_temp(y_true, y_pred).numpy(), 0.38383293)