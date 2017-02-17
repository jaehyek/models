# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory
import numpy as np
import math
import time

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
  'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
  'max_num_batches', None,
  'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
  'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
  'checkpoint_path', '/tmp/tfmodel/',
  'The directory where the model was written to or an absolute path to a '
  'checkpoint file.')

tf.app.flags.DEFINE_string(
  'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
  'num_preprocessing_threads', 4,
  'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
  'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
  'dataset_name_suffix', None, 'The suffix of name of the dataset to load.')

tf.app.flags.DEFINE_string(
  'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
  'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
  'labels_offset', 0,
  'An offset for the labels in the dataset. This flag is primarily used to '
  'evaluate the VGG and ResNet architectures which do not use a background '
  'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
  'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
  'preprocessing_name', None, 'The name of the preprocessing to use. If left '
                              'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
  'moving_average_decay', None,
  'The decay to use for the moving average.'
  'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
  'eval_image_size', None, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def put_kernels_on_grid(kernel, pad=1):
  '''Visualize conv. features as an image (mostly for the 1st layer).
  Place kernel into a grid, with some paddings between adjacent filters.

  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    (grid_Y, grid_X):  shape of the grid. Require: NumKernels == grid_Y * grid_X
                         User is responsible of how to break into two multiples.
    pad:               number of black pixels around each filter (between them)

  Return:
    Tensor of shape [(Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels, 1].
  '''

  def factorization(n):
    for i in range(int(math.sqrt(float(n))), 0, -1):
      if n % i == 0:
        if i == 1: print('Who would enter a prime number of filters')
        return (i, int(n / i))

  (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)

  kernel1 = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x1 = tf.pad(kernel1, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel1.get_shape()[0] + 2 * pad
  X = kernel1.get_shape()[1] + 2 * pad

  channels = kernel1.get_shape()[2]

  # put NumKernels to the 1st dimension
  x2 = tf.transpose(x1, (3, 0, 1, 2))
  # organize grid on Y axis
  x3 = tf.reshape(x2, tf.pack([grid_X, Y * grid_Y, X, channels]))  # 3

  # switch X and Y axes
  x4 = tf.transpose(x3, (0, 2, 1, 3))
  # organize grid on X axis
  x5 = tf.reshape(x4, tf.pack([1, X * grid_X, Y * grid_Y, channels]))  # 3

  # back to normal order (not combining with the next step for clarity)
  x6 = tf.transpose(x5, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x7 = tf.transpose(x6, (3, 0, 1, 2))

  # scale to [0, 255] and convert to uint8
  return tf.image.convert_image_dtype(x7, dtype=tf.uint8)


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')
  times = {}
  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    start = time.time()
    tf_global_step = slim.get_or_create_global_step()
    times['global_step'] = time.time() - start

    ######################
    # Select the dataset #

    start = time.time()
    dataset = dataset_factory.get_dataset(
      FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir, suffix=FLAGS.dataset_name_suffix)
    times['get_dataset'] = time.time() - start

    ####################
    # Select the model #
    ####################
    start = time.time()
    network_fn = nets_factory.get_network_fn(
      FLAGS.model_name,
      num_classes=(dataset.num_classes - FLAGS.labels_offset),
      is_training=False)
    times['select_model'] = time.time() - start

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    start = time.time()
    provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      shuffle=False,
      common_queue_capacity=2 * FLAGS.batch_size,
      common_queue_min=FLAGS.batch_size)
    times['get_provider'] = time.time() - start
    start = time.time()
    [image] = provider.get(['image'])
    times['get_image'] = time.time() - start

    #####################################
    # Select the preprocessing function #
    #####################################
    start = time.time()
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
      preprocessing_name,
      is_training=False)
    times['get_preprocessing'] = time.time() - start

    eval_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    start = time.time()
    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)
    times['preprocessing'] = time.time() - start

    start = time.time()
    images = tf.train.batch(
      [image],
      batch_size=FLAGS.batch_size,
      num_threads=FLAGS.num_preprocessing_threads,
      capacity=5 * FLAGS.batch_size)
    times['get_batch'] = time.time() - start

    start = time.time()
    tf.image_summary('test_images', images, FLAGS.batch_size)
    times['image_summary'] = time.time() - start

    ####################
    # Define the model #
    ####################
    start = time.time()
    logits, _ = network_fn(images)
    times['do_network'] = time.time() - start

    # with tf.variable_scope('resnet_v2_152/block1/unit_1/bottleneck_v2/conv1', reuse=True):
    #   weights = tf.get_variable('weights')
    #   kernel_transposed = put_kernels_on_grid(weights)
    # scale weights to [0 1], type is still float
    # x_min = tf.reduce_min(weights)
    # x_max = tf.reduce_max(weights)
    # kernel_0_to_1 = (weights - x_min) / (x_max - x_min)
    #
    # # to tf.image_summary format [batch_size, height, width, channels]
    # kernel_transposed = tf.transpose(kernel_0_to_1, [3, 0, 1, 2])

    # this will display random 3 filters from the 64 in conv1
    # tf.image_summary('conv1/filters', kernel_transposed, max_images=50)

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
        slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    if len(logits.get_shape()) == 4:
      logits = tf.reshape(logits, [int(logits.get_shape()[0]), -1])

    softmax = tf.nn.softmax(logits)
    # predictions = tf.argmax(logits, 1)

    # Define the metrics:
    # names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
    # 'Predictions': predictions,
    # 'Predictions': slim.metrics.streaming_accuracy(predictions, labels),
    # 'Predictions@5': slim.metrics.streaming_recall_at_k(
    #   logits, labels, 5),
    # })

    # Print the summaries to screen.
    # for name, value in names_to_values.iteritems():
    #   summary_name = 'eval/%s' % name
    #   op = tf.scalar_summary(summary_name, value, collections=[])
    #   op = tf.Print(op, [value], summary_name)
    #   tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    start = time.time()
    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path
    times['load_checkpoint'] = time.time() - start

    tf.logging.info('Evaluating %s' % checkpoint_path)
    # evaluate_loop
    start = time.time()
    from tensorflow.contrib.framework.python.ops import variables
    from tensorflow.core.protobuf import saver_pb2
    from tensorflow.python.training import saver as tf_saver
    from tensorflow.python.framework import ops
    from tensorflow.python.training import supervisor
    saver = tf_saver.Saver(
      variables_to_restore or variables.get_variables_to_restore(),
      write_version=saver_pb2.SaverDef.V1)
    sv = supervisor.Supervisor(graph=ops.get_default_graph(),
                               logdir=FLAGS.eval_dir,
                               summary_op=None,
                               summary_writer=None,
                               global_step=None,
                               saver=None)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    saver.restore(sess, checkpoint_path)
    sv.start_queue_runners(sess)
    final_op_value = sess.run(logits)
    # final_op_value = slim.evaluation.evaluate_once(
    #   master=FLAGS.master,
    #   checkpoint_path=checkpoint_path,
    #   logdir=FLAGS.eval_dir,
    #   num_evals=num_batches,
    #   final_op=[softmax, logits],
    #   # eval_op=names_to_updates.values(),
    #   variables_to_restore=variables_to_restore)
    times['exec'] = time.time() - start

    print(final_op_value[1].shape)
    result_predict = np.reshape(final_op_value[1], (FLAGS.batch_size, final_op_value[1].shape[-1]))
    # print(final_op_value)
    print(result_predict)
    print(np.argsort(result_predict[:, 1])[-5:])
  print(times)


if __name__ == '__main__':
  tf.app.run()
