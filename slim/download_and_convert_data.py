# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts a particular dataset.

Usage:
```shell

$ python download_and_convert_data.py \
    --dataset_name=mnist \
    --dataset_dir=/tmp/mnist

$ python download_and_convert_data.py \
    --dataset_name=cifar10 \
    --dataset_dir=/tmp/cifar10

$ python download_and_convert_data.py \
    --dataset_name=flowers \
    --dataset_dir=/tmp/flowers

$ python download_and_convert_data.py \
    --dataset_name=koreans \
    --dataset_dir=/tmp/koreans
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import download_and_convert_cifar10
from datasets import download_and_convert_flowers
from datasets import download_and_convert_mnist
from datasets import download_and_convert_koreans
from datasets import download_and_convert_apparel
from datasets import download_and_convert_apparelv
from datasets import download_and_convert_apparelv_dabainsang
from datasets import download_and_convert_apparelv_dlsdl113
from datasets import download_and_convert_apparelv_binary
from datasets import download_and_convert_apparelv_binary_without_dummy
from datasets import download_and_convert_apparelv_dabainsang_without_dummy

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
  'dataset_name',
  None,
  'The name of the dataset to convert, one of "cifar10", "flowers", "mnist", "koreans".')

tf.app.flags.DEFINE_string(
  'dataset_dir',
  None,
  'The directory where the output TFRecords and temporary files are saved.')

tf.app.flags.DEFINE_boolean(
  'custom_binary_validation',
  False,
  'validation data label')

tf.app.flags.DEFINE_string(
  'custom_binary_validation_label',
  '1',
  'validation data label')

tf.app.flags.DEFINE_float(
  'custom_binary_validation_ratio',
  0.5,
  'validation data ratio')

tf.app.flags.DEFINE_string(
  'output_suffix',
  None,
  'validation data ratio')

tf.app.flags.DEFINE_boolean(
  'test_data',
  False,
  'validation data ratio')

tf.app.flags.DEFINE_boolean(
  'is_other_dir',
  False,
  'validation data ratio')


def main(_):
  if not FLAGS.dataset_name:
    raise ValueError('You must supply the dataset name with --dataset_name')
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  if FLAGS.dataset_name == 'cifar10':
    download_and_convert_cifar10.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'flowers':
    download_and_convert_flowers.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'mnist':
    download_and_convert_mnist.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'koreans':
    download_and_convert_koreans.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'apparel':
    download_and_convert_apparel.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'apparelv':
    download_and_convert_apparelv.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'apparelv_dabainsang':
    download_and_convert_apparelv_dabainsang.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'apparelv_dabainsang_without_dummy':
    download_and_convert_apparelv_dabainsang_without_dummy.run(FLAGS.dataset_dir,
                                                               FLAGS.custom_binary_validation,
                                                               FLAGS.custom_binary_validation_label,
                                                               FLAGS.custom_binary_validation_ratio,
                                                               FLAGS.output_suffix,
                                                               FLAGS.is_other_dir)
  elif FLAGS.dataset_name == 'apparelv_dlsdl113':
    download_and_convert_apparelv_dlsdl113.run(FLAGS.dataset_dir)
  elif FLAGS.dataset_name == 'apparelv_binary':
    if FLAGS.test_data:
      download_and_convert_apparelv_binary.run_for_test(FLAGS.dataset_dir)
    else:
      download_and_convert_apparelv_binary.run(FLAGS.dataset_dir, FLAGS.custom_binary_validation,
                                               FLAGS.custom_binary_validation_label,
                                               FLAGS.custom_binary_validation_ratio,
                                               FLAGS.output_suffix)
  elif FLAGS.dataset_name == 'apparelv_binary_without_dummy':
    download_and_convert_apparelv_binary_without_dummy.run(FLAGS.dataset_dir, FLAGS.custom_binary_validation,
                                                           FLAGS.custom_binary_validation_label,
                                                           FLAGS.custom_binary_validation_ratio,
                                                           FLAGS.output_suffix)
  else:
    raise ValueError(
      'dataset_name [%s] was not recognized.' % FLAGS.dataset_dir)


if __name__ == '__main__':
  tf.app.run()
