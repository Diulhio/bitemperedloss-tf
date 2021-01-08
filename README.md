# bitemperedloss-tf
Simple Bi-Tempered Loss implementation for Tensorflow 2.0

This is not an officially supported Google product.

Overview of the method is here: [link Google AI Blogpost](https://ai.googleblog.com/2019/08/bi-tempered-logistic-loss-for-training.html)

- Original repo: https://github.com/google/bi-tempered-loss
- Adapted from: https://github.com/mlpanda/bi-tempered-loss-pytorch
- Paper: "Robust Bi-Tempered Logistic Loss Based on Bregman Divergences" (https://arxiv.org/abs/1906.03361)

To test the loss function with similar values to the test in the original repo, run the following file:
```
python test_loss.py
```

## Requeriments

Tensorflow >= 2.0

## Usage in Tensorflow

Call as any loss function in Tensorflow, for example:

```
import tensorflow as tf
from tf_bi_tempered_loss import BiTemperedLogisticLoss

pretrained_model = tf.keras.applications.VGG16()
pretrained_model.compile(optimizer='adam', loss=BiTemperedLogisticLoss(t1=1.0, t2=1.0), metrics=['accuracy'])
pretrained_model.fit(your_params)
```
