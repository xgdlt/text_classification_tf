# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random

from util import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, input_ids, label_ids, tokens):
        self.input_ids = input_ids
        self.label_ids = label_ids
        self.tokens = tokens

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "input_ids: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.input_ids]))
        s += "label_ids: %s\n" % (" ".join([str(x) for x in self.label_ids]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = instance.input_ids
        label_ids = instance.label_ids

        if len(input_ids) > max_seq_length:
            input_ids = input_ids[0:max_seq_length]

        while len(input_ids) < max_seq_length:
            input_ids.append(0)

        assert len(input_ids) == max_seq_length

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["label_ids"] = create_int_feature(label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    tf.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_instances(input_files, tokenizer, max_seq_length):
    """Create `TrainingInstance`s from raw text."""
    all_documents = []

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.

    for input_file in input_files:
        all_documents.append([])
        with tf.gfile.GFile(input_file, "r") as reader:
            for line in reader.readlines():
                line = tokenization.convert_to_unicode(line)
                lst = line.strip().split("\t")
                if len(lst) < 2:
                    print("error line: ", line)
                tokens = tokenizer.tokenize(lst[0])
                if tokens:
                    all_documents[-1].append([tokens,lst[1]])
    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    random.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab.keys())
    instances = []
    for document_index in range(len(all_documents)):
        instances.extend(
            create_instances_from_document(
                all_documents, document_index, max_seq_length,
                 vocab_words))

    random.shuffle(instances)
    return instances


def create_instances_from_document(
        documents,  max_seq_length,
        vocab_words,labels):
    """Creates `TrainingInstance`s for a single document."""

    instances = []



    i = 0
    for sentence, label in documents:
        tokens = tokenization.tokenize_word(sentence, True)
        input_ids = tokenization.convert_by_vocab(vocab_words, tokens)
        label_ids = labels.index(label)
        instance = TrainingInstance(
            input_ids, label_ids, tokens
        )
        instances.append(instance)


    return instances


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(tf.gfile.Glob(input_pattern))

    tf.logging.info("*** Reading from input files ***")
    for input_file in input_files:
        tf.logging.info("  %s", input_file)

    instances = create_training_instances(
        input_files, tokenizer, FLAGS.max_seq_length)

    output_files = FLAGS.output_file.split(",")
    tf.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.logging.info("  %s", output_file)

    write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                    FLAGS.max_predictions_per_seq, output_files)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("output_file")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
