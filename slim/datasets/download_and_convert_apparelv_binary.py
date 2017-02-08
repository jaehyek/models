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
r"""Downloads and converts Koreans data to TFRecords of TF-Example protos.

This module downloads the Koreans data, uncompresses it, reads the files
that make up the Koreans data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

# The URL where the Koreans data can be downloaded.
_DATA_URL = 'http://122.38.108.102/file/data/apparel/images.tgz'

# The number of images in the validation set.
_NUM_VALIDATION = 6387

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes Grayscale JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _get_filenames_and_classes(dataset_dir, dataset_name):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  koreans_root = os.path.join(dataset_dir, dataset_name)
  directories = []
  class_names = []
  for filename in os.listdir(koreans_root):
    path = os.path.join(koreans_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []
  for directory in directories:
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames.append(path)

  return photo_filenames, sorted(class_names)


def _get_filenames_and_classes_by_label(dataset_dir, dataset_name):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  koreans_root = os.path.join(dataset_dir, dataset_name)
  directories = []
  class_names = []
  for filename in os.listdir(koreans_root):
    path = os.path.join(koreans_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = {}
  for directory in directories:
    label_name = os.path.basename(directory)
    photo_filenames[label_name] = []
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      photo_filenames[label_name].append(path)

  return photo_filenames, sorted(class_names)


def _get_filenames(dataset_dir):
  """Returns a list of filenames and inferred class names.

  Args:
    dataset_dir: A directory containing a set of subdirectories representing
      class names. Each subdirectory should contain PNG or JPG encoded images.

  Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
  """
  photo_filenames = []
  for filename in os.listdir(dataset_dir):
    path = os.path.join(dataset_dir, filename)
    photo_filenames.append(path)

  return sorted(photo_filenames)


def _get_dataset_filename(dataset_dir, split_name, shard_id, output_suffix=None):
  if output_suffix:
    output_filename = 'apparelv_%s_%s_%05d-of-%05d.tfrecord' % (
      output_suffix, split_name, shard_id, _NUM_SHARDS)
  else:
    output_filename = 'apparelv%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _get_dataset_filename_for_test(dataset_dir, shard_id):
  output_filename = 'apparelv_test_%05d-of-%05d.tfrecord' % (shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir, output_suffix):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id, output_suffix)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d, %s' % (
              i + 1, len(filenames), shard_id, filenames[i]))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            if sys.version_info[0] == 3:
              example = dataset_utils.image_to_tfexample(
                image_data, 'jpg'.encode(), height, width, class_id)
            else:
              example = dataset_utils.image_to_tfexample(
                image_data, 'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _convert_dataset_for_test(filenames, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  num_per_shard = int(math.ceil(len(filenames) / float(1)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(1):
        output_filename = _get_dataset_filename_for_test(dataset_dir, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d, %s' % (
              i + 1, len(filenames), shard_id, filenames[i]))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'r').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            if sys.version_info[0] == 3:
              example = dataset_utils.image_to_tfexample_for_test(
                image_data, 'jpg'.encode(), height, width)
            else:
              example = dataset_utils.image_to_tfexample_for_test(
                image_data, 'jpg', height, width)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _clean_up_temporary_files(dataset_dir, dataset_name):
  """Removes temporary files used to create the dataset.

  Args:
    dataset_dir: The directory where the temporary files are stored.
  """
  filename = _DATA_URL.split('/')[-1]
  filepath = os.path.join(dataset_dir, filename)
  tf.gfile.Remove(filepath)

  tmp_dir = os.path.join(dataset_dir, dataset_name)
  tf.gfile.DeleteRecursively(tmp_dir)


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
        dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(dataset_dir, custom_binary_validation=None, custom_binary_validation_label=None,
        custom_binary_validation_ratio=None,
        output_suffix=None):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return
  random.seed(_RANDOM_SEED)
  # dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
  if custom_binary_validation:
    tmp_photo_filenames, class_names = _get_filenames_and_classes_by_label(dataset_dir, 'apparelv_binary')
    if not custom_binary_validation_ratio:
      custom_binary_validation_ratio = 0.
    if custom_binary_validation_ratio > 1:
      custom_binary_validation_ratio = 1.
    validation_filenames = []
    training_filenames = []
    for key in tmp_photo_filenames:
      if key == custom_binary_validation_label:
        ratio = custom_binary_validation_ratio
      else:
        ratio = 1. - custom_binary_validation_ratio

      random.shuffle(tmp_photo_filenames[key])
      training_filenames += tmp_photo_filenames[key][int(_NUM_VALIDATION * ratio):]
      print(key, len(tmp_photo_filenames[key][:int(_NUM_VALIDATION * ratio)]))
      validation_filenames += tmp_photo_filenames[key][:int(_NUM_VALIDATION * ratio)]
  else:
    photo_filenames, class_names = _get_filenames_and_classes(dataset_dir, 'apparelv_binary')

    # Divide into train and test:
    print("Now let's start converting the Koreans dataset!")
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[_NUM_VALIDATION:]
    validation_filenames = photo_filenames[:_NUM_VALIDATION]

  class_names_to_ids = dict(zip(class_names, range(len(class_names))))
  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, class_names_to_ids,
                   dataset_dir, output_suffix)
  _convert_dataset('validation', validation_filenames, class_names_to_ids,
                   dataset_dir, output_suffix)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  if output_suffix:
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir, 'labels_' + output_suffix + '.txt')
  else:
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  # _clean_up_temporary_files(dataset_dir, 'apparel')
  print('\nFinished converting the Koreans dataset!')


def run_for_test(dataset_dir):
  """Runs the download and conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  photo_filenames = _get_filenames(dataset_dir)
  # Divide into train and test:
  print("Now let's start converting the Koreans dataset!")
  _convert_dataset_for_test(photo_filenames, dataset_dir)

  # _clean_up_temporary_files(dataset_dir, 'apparel')
  print('\nFinished converting the test dataset!')
