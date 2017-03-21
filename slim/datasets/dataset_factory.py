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
"""A factory-pattern class which returns classification image/label pairs."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from slim.datasets import cifar10
from slim.datasets import flowers
from slim.datasets import imagenet
from slim.datasets import mnist
from slim.datasets import koreans
from slim.datasets import apparel
from slim.datasets import apparelv
from slim.datasets import apparelv_dabainsang
from slim.datasets import apparelv_dabainsang_without_dummy
from slim.datasets import apparelv_dlsdl113
from slim.datasets import apparelv_binary
from slim.datasets import apparelv_binary_without_dummy
from slim.datasets import common

datasets_map = {
    'cifar10': cifar10,
    'flowers': flowers,
    'imagenet': imagenet,
    'mnist': mnist,
    'koreans': koreans,
    'apparel': apparel,
    'apparelv': apparelv,
    'apparelv_dabainsang': apparelv_dabainsang,
    'apparelv_dabainsang_without_dummy': apparelv_dabainsang_without_dummy,
    'apparelv_dlsdl113': apparelv_dlsdl113,
    'apparelv_binary': apparelv_binary,
    'apparelv_binary_without_dummy': apparelv_binary_without_dummy,
    'common': common,
}


def get_dataset(name, split_name, dataset_dir, file_pattern=None, reader=None, suffix=None):
    """Given a dataset name and a split_name returns a Dataset.

  Args:
    name: String, the name of the dataset.
    split_name: A train/test split name.
    dataset_dir: The directory where the dataset files are stored.
    file_pattern: The file pattern to use for matching the dataset source files.
    reader: The subclass of tf.ReaderBase. If left as `None`, then the default
      reader defined by each dataset is used.

  Returns:
    A `Dataset` class.

  Raises:
    ValueError: If the dataset `name` is unknown.
  """
    if name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % name)
    if suffix:
        return datasets_map[name].get_split(
            split_name,
            dataset_dir,
            file_pattern,
            reader,
            suffix=suffix)
    else:
        return datasets_map[name].get_split(
            split_name,
            dataset_dir,
            file_pattern,
            reader)
