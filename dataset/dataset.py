#! -*- coding: utf-8 -*-
# 预训练语料构建
"""
Author:
    LiTeng 1471356861@qq.com
"""

import os
import numpy as np
import tensorflow as tf
from dataset.tokenizer import Tokenizer
import glob


class TrainingDataset(object):
    """预训练数据集生成器
    """
    def __init__(self, tokenizer, labels =[],sequence_length=512):
        """参数说明：
            tokenizer必须是bert4keras自带的tokenizer类；
        """
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.labels = labels
        self.label_to_id = {}
        for i,label in enumerate(labels):
            self.label_to_id[label] = i
        self.token_pad_id = tokenizer._token_pad_id
        self.token_cls_id = tokenizer._token_cls_id
        self.token_sep_id = tokenizer._token_sep_id
        self.token_mask_id = tokenizer._token_mask_id
        self.token_pad_id = tokenizer._token_pad_id
        self.vocab_size = tokenizer._vocab_size

    def padding(self, sequence, padding_value=None):
        """对单个序列进行补0
        """
        if padding_value is None:
            padding_value = self.token_pad_id

        sequence = sequence[:self.sequence_length]
        padding_length = self.sequence_length - len(sequence)
        return sequence + [padding_value] * padding_length

    def sentence_process(self, text_a,target,text_b=None,):
        """单个文本的处理函数
               流程：分词，然后转id，按照mask_rate构建全词mask的序列
                     来指定哪些token是否要被mask
               """
        word_tokens = self.tokenizer.tokenize(text=text_a)
        token_ids = self.tokenizer.tokens_to_ids(word_tokens)

        token_ids = token_ids[0:self.sequence_length]

        # 如果长度即将溢出
        while len(token_ids) < self.sequence_length:
            # 插入终止符，并padding
            token_ids.append(self.token_pad_id)

        target_ids = []
        for label in target.split(" "):
            target_ids.append(self.label_to_id.get(label,len(self.labels)))


        return [token_ids,target_ids]


    def paragraph_process(self, corpus):
        """单个段落（多个文本）的处理函数
        说明：texts是单句组成的list；
        做法：不断塞句子，直到长度最接近sequence_length，然后padding。
        """
        instances = []
        for item in corpus:
            # 处理单个句子
            #print(item)
            text, target = item
            instance = self.sentence_process(text,target)

            instances.append(instance)
        return instances

    def tfrecord_serialize(self, instances):
        """转为tfrecord的字符串，等待写入到文件
        """
        instance_keys = ['token_ids', 'label_ids']
        def create_feature(x):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=x))

        serialized_instances = []
        for instance in instances:
            features = {
                k: create_feature(v)
                for k, v in zip(instance_keys, instance)
            }
            tf_features = tf.train.Features(feature=features)
            tf_example = tf.train.Example(features=tf_features)
            serialized_instance = tf_example.SerializeToString()
            serialized_instances.append(serialized_instance)

        return serialized_instances


    def process(self, corpus, record_name):
        """处理输入语料（corpus），最终转为tfrecord格式（record_name）
        自带多进程支持，如果cpu核心数多，请加大workers和max_queue_size。
        """
        writer = tf.io.TFRecordWriter(record_name)
        globals()['count'] = 0
        instances = self.paragraph_process(corpus)
        serialized_instances = self.tfrecord_serialize(instances)
        globals()['count'] += len(serialized_instances)
        for serialized_instance in serialized_instances:
            writer.write(serialized_instance)

        writer.close()
        print('write %s examples into %s' % (globals()['count'], record_name))

    @staticmethod
    def load_tfrecord(record_names,sequence_length, batch_size,label_length=None):
        """加载处理成tfrecord格式的语料
        """
        """给原方法补上parse_function
              """

        def parse_function(serialized):
            features = {
                'token_ids': tf.io.FixedLenFeature([sequence_length], tf.int64),
                'label_ids': tf.io.FixedLenFeature([], tf.int64),
            }
            features = tf.io.parse_single_example(serialized, features)
            token_ids = features['token_ids']
            label_ids = features['label_ids']

            x = {
                'token_ids': token_ids
            }
            y = {
                'label_ids': label_ids
            }
            return token_ids, label_ids

        if not isinstance(record_names, list):
            record_names = [record_names]

        dataset = tf.data.TFRecordDataset(record_names)  # 加载
        dataset = dataset.map(parse_function)  # 解析
        dataset = dataset.repeat()  # 循环
        dataset = dataset.shuffle(batch_size * 1000)  # 打乱
        dataset = dataset.batch(batch_size)  # 成批

        return dataset


if __name__ == '__main__':



    model = 'roberta'
    sequence_length = 512
    workers = 40
    max_queue_size = 4000
    dict_path = '../conf/vocab.txt'
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    def some_texts():
        filenames = glob.glob('D:\\一键分类\\涉政\\train.txt')
        np.random.shuffle(filenames)
        texts =  []
        for filename in filenames:
            with open(filename,encoding="utf8") as f:
                for l in f:
                    lst = l.strip().split("\t")
                    if len(lst) < 2:
                        continue
                    texts.append([lst[0],lst[1]])
        return texts[0:1000]

    assert model in ['roberta', 'gpt']  # 判断是否支持的模型类型

    if model == 'roberta':

        '''
        import jieba_fast as jieba
        jieba.initialize()

        def word_segment(text):
            return jieba.lcut(text)
       '''
        TD = TrainingDataset(tokenizer, labels=["正常","涉政负面"], sequence_length=sequence_length)

        TD.process(
                corpus=some_texts(),
                record_name='D:\\一键分类\\涉政\\corpus.tfrecord'
            )

    elif model == 'gpt':

        TD = TrainingDataset(tokenizer, sequence_length=sequence_length)

        TD.process(
            corpus=some_texts(),
            record_name='../corpus_tfrecord/corpus.tfrecord',
            workers=workers,
            max_queue_size=max_queue_size,
        )
