# Optimizer correctness reference implementations for
# Tests/TensorFlowTests/OptimizerTests.swift.

# Tested with:
# - tensorflow==2.2.0rc0
# - tensorflow-addons==0.8.3

import tensorflow as tf
from tensorflow_addons.optimizers import RectifiedAdam

def test_optimizer(optimizer, step_count=1000):
  weight = tf.Variable([[0.8]], dtype=tf.dtypes.float32)
  weight_grad = tf.Variable([[0.1]], dtype=tf.dtypes.float32)
  bias = tf.Variable([0.8], dtype=tf.dtypes.float32)
  bias_grad = tf.Variable([0.2], dtype=tf.dtypes.float32)
  grads_and_vars = list(zip([weight_grad, bias_grad], [weight, bias]))
  for i in range(step_count):
    optimizer.apply_gradients(grads_and_vars)

  print(optimizer._name)
  print('- weight', weight.read_value().numpy())
  print('- bias', bias.read_value().numpy())

test_optimizer(RectifiedAdam(lr=1e-3, epsilon=1e-8))
