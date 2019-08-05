#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
import random
import numpy as np
import tensorflow as tf
import collections
from util import tokenization



def input_fn_builder(input_files,
                     max_seq_length,
                     lable_size,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "label":
            tf.FixedLenFeature([lable_size], tf.float32)
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example



def read_dcsv_ata(data_file, batch_size,max_seq_length,
                     lable_size,is_training):

    data_files = tf.gfile.Glob(data_file)
    file_queue = tf.train.string_input_producer(data_files, shuffle=True)
    reader = tf.TextLineReader()
    _,value = reader.read(file_queue)
    record_defaults = [[0] * max_seq_length, [0] * lable_size]
    input_id, label_id = tf.decode_csv(value,record_defaults)
    example_enqueue = tf.RandomShuffleQueue()

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def get_filelist(file_path, filename, output_dir,batch_size=64):
    train_file = os.path.join(file_path, filename)
    if not os.path.exists(train_file):
        raise ValueError(train_file + "is not exists")
    output_path = os.path.join(file_path, output_dir)
    if os.path.exists(output_path):
        files = os.listdir(output_path)  # 列出目录下的文件
        for file in files:
            os.remove(file)  # 删除文件
    else:
        os.makedirs(output_path)

    lines = open(train_file).readlines();
    random.shuffle(lines)
    filecnt = 0
    f = open("%s/train%d")

def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f

def file_based_convert_examples_to_features(examples, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""
  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
      if ex_index % 10000 == 0:
          tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
      input_ids, label_ids = example


      features = collections.OrderedDict()
      features["input_ids"] = create_int_feature(input_ids)
      features["label_ids"] = create_int_feature(label_ids)

      tf_example = tf.train.Example(features=tf.train.Features(feature=features))
      writer.write(tf_example.SerializeToString())

def read_my_file_format(filename_queue,max_seq_length, num_classes):
  reader = tf.TFRecordReader()
  key, record_string = reader.read(filename_queue)
  name_to_features = {
      "input_ids":
          tf.FixedLenFeature([max_seq_length], tf.int64),
      "label_ids":
          tf.FixedLenFeature([num_classes], tf.float32)
  }
  example, label = tf.parse_single_example(record_string, name_to_features)
  #processed_example = some_processing(example)
  return example, label

def input_pipeline(filenames, batch_size,max_seq_length, num_classes, num_epochs=1):
  filename_queue = tf.train.string_input_producer(
      filenames, num_epochs=num_epochs, shuffle=True)
  example, label = read_my_file_format(filename_queue,max_seq_length, num_classes)
  # min_after_dequeue defines how big a buffer we will randomly sample
  #   from -- bigger means better shuffling but slower start up and more
  #   memory used.
  # capacity must be larger than min_after_dequeue and the amount larger
  #   determines the maximum we will prefetch.  Recommendation:
  #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
  min_after_dequeue = 10000
  capacity = min_after_dequeue + 3 * batch_size
  example_batch, label_batch = tf.train.shuffle_batch(
      [example, label], batch_size=batch_size, capacity=capacity,
      min_after_dequeue=min_after_dequeue)
  return example_batch, label_batch


def read_csv(input_file):
    """Reads a tab separated value file."""
    lines = []
    for line in open(input_file, "r", encoding="utf-8"):
        #print(line)
        if line.strip() in ["", " "]:
            continue
        lst = line.strip().split("\t")
        lines.append(lst)
    return lines

def write_csv(output_file, lines):
    fw = open(output_file, "a")
    for line in lines:
        fw.write("%s\n"%("\t".join(line)))
    return

def write_file(output_file, lines):
    fw = open(output_file, "a", encoding="utf-8")
    for line in lines:
        fw.write("%s\n"%(line))
    return

def get_data_examples(input_file,example_index=0, label_index=1, segger=False):
    """Reads a tab separated value file."""
    print("example_index = ", example_index, "label_index = ", label_index)
    lines = read_csv(input_file)
    examples = []
    labels = []
    for  line in lines:
        if len(line) < example_index+1 :
            examples.append("")
            labels.append("")
            continue
        #print(line)
        example = line[example_index]
        #print(example)
        if segger:
            example = tokenization.tokenize_word(example)
        #print(example)
        examples.append(" ".join(example))

        if len(line) < label_index+1:
            labels.append(" ")
        else:
            label = line[label_index]
            labels.append(label)
    return examples, labels


def get_predict_data(input_file,example_index=0, segger=False):
    """Reads a tab separated value file."""
    print("predict example_index = ", example_index)
    lines = read_csv(input_file)
    examples = []
    for  line in lines:
        if len(line) < example_index+1 :
            continue
        #print(line)
        example = line[example_index]
        #print(example)
        if segger:
            example = tokenization.tokenize_word(example)
        #print(example)
        examples.append(" ".join(example))
    return examples

def get_index(input_file):
    """Reads a tab separated value file."""
    lines = read_csv(input_file)
    indexs = {}
    index = 0
    for line in lines:
        if len(line) < 1:
            continue
        indexs[line[0]] = index
        index += 1
    return indexs

def convert_to_ids_by_vocab(max_seq_length,vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  outputs = []
  for item in items:
      output = []
      words = item.split(" ")
      for word in words:
          if word in vocab:
              output.append(vocab.get(word))
          else:
              output.append(vocab.get("[UNK]"))
      output = output[0:max_seq_length]
      while len(output) < max_seq_length:
          output.append(vocab.get("[PAD]"))
      outputs.append(output)
  return outputs

def convert_to_one_hots(label2index, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  outputs = []
  for item in items:
      output = [0] * len(label2index)
      labels = item.split(" ")
      for label in labels:
          label_index = label2index.get(label, -1)
          if label_index >= 0:
              output[label_index] = 1
      outputs.append(output)
  return outputs

def convert_to_label_num(possibilitys):
  """Converts a sequence of [tokens|ids] using the vocab."""
  label = []
  for index, values in enumerate(possibilitys):
      value = values.index(max(values))
      label.append(value)

  return label