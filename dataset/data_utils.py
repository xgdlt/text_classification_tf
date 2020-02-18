#! -*- coding: utf-8 -*-
# 预训练语料构建

import os
os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import numpy as np
import tensorflow as tf
from dataset.snippets import parallel_apply
from dataset.backend import K


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
        self.vocab_size = tokenizer._vocab_size

    def padding(self, sequence, padding_value=None):
        """对单个序列进行补0
        """
        if padding_value is None:
            padding_value = self.token_pad_id

        sequence = sequence[:self.sequence_length]
        padding_length = self.sequence_length - len(sequence)
        return sequence + [padding_value] * padding_length

    def sentence_process(self, text_a,targets,text_b=None):
        """单个文本的处理函数
               流程：分词，然后转id，按照mask_rate构建全词mask的序列
                     来指定哪些token是否要被mask
               """
        token_ids = []
        word_tokens = self.tokenizer.tokenize(text=text_a)
        word_token_ids = self.tokenizer.tokens_to_ids(word_tokens)
        token_ids.extend(word_token_ids)

        target_ids = []
        for target in targets:
            target_ids.append(self.label_to_id.get(target,len(self.labels)))


        return [token_ids,target_ids]


    def paragraph_process(self, texts, padding_id):
        """单个段落（多个文本）的处理函数
        说明：texts是单句组成的list；starts是每个instance的起始id；
              ends是每个instance的终止id；paddings是每个instance的填充id。
        做法：不断塞句子，直到长度最接近sequence_length，然后padding。
        """
        instances = []
        for text in texts:
            # 处理单个句子
            token_ids,target_ids = self.sentence_process(text)
            token_ids = token_ids[0:self.sequence_length]

            # 如果长度即将溢出
            while len(token_ids)  < self.sequence_length:
                # 插入终止符，并padding
                token_ids.append(padding_id)

            # 样本续接
            instances.append(token_ids)

        return instances

    def tfrecord_serialize(self, instances, instance_keys):
        """转为tfrecord的字符串，等待写入到文件
        """
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

    def process(self, corpus, record_name, workers=8, max_queue_size=2000):
        """处理输入语料（corpus），最终转为tfrecord格式（record_name）
        自带多进程支持，如果cpu核心数多，请加大workers和max_queue_size。
        """
        writer = tf.io.TFRecordWriter(record_name)
        globals()['count'] = 0

        def write_to_tfrecord(serialized_instances):
            globals()['count'] += len(serialized_instances)
            for serialized_instance in serialized_instances:
                writer.write(serialized_instance)

        def paragraph_process(texts):
            instances = self.paragraph_process(texts)
            serialized_instances = self.tfrecord_serialize(instances)
            return serialized_instances

        parallel_apply(
            func=paragraph_process,
            iterable=corpus,
            workers=workers,
            max_queue_size=max_queue_size,
            callback=write_to_tfrecord,
        )

        writer.close()
        print('write %s examples into %s' % (globals()['count'], record_name))

    @staticmethod
    def load_tfrecord(record_names, batch_size, parse_function):
        """加载处理成tfrecord格式的语料
        """
        if not isinstance(record_names, list):
            record_names = [record_names]

        dataset = tf.data.TFRecordDataset(record_names)  # 加载
        dataset = dataset.map(parse_function)  # 解析
        dataset = dataset.repeat()  # 循环
        dataset = dataset.shuffle(batch_size * 1000)  # 打乱
        dataset = dataset.batch(batch_size)  # 成批

        return dataset


class TrainingDatasetRoBERTa(TrainingDataset):
    """预训练数据集生成器（RoBERTa模式）
    """
    def __init__(self,
                 tokenizer,
                 word_segment,
                 mask_rate=0.15,
                 sequence_length=512):
        """参数说明：
            tokenizer必须是bert4keras自带的tokenizer类；
            word_segment是任意分词函数。
        """
        super(TrainingDatasetRoBERTa, self).__init__(tokenizer, sequence_length)
        self.word_segment = word_segment
        self.mask_rate = mask_rate

    def token_process(self, token_id):
        """以80%的几率替换为[MASK]，以10%的几率保持不变，
        以10%的几率替换为一个随机token。
        """
        rand = np.random.random()
        if rand <= 0.8:
            return self.token_mask_id
        elif rand <= 0.9:
            return token_id
        else:
            return np.random.randint(0, self.vocab_size)

    def sentence_process(self, text_a, text_b = None):
        """单个文本的处理函数
        流程：分词，然后转id，按照mask_rate构建全词mask的序列
              来指定哪些token是否要被mask
        """
        token_ids = []

        word_tokens = self.tokenizer.tokenize(text=text_a)
        word_token_ids = self.tokenizer.tokens_to_ids(word_tokens)
        token_ids.extend(word_token_ids)


        return token_ids

    def paragraph_process(self, texts):
        """给原方法补上starts、ends、paddings
        """
        starts = [self.token_cls_id, 0]
        ends = [self.token_sep_id, 0]
        paddings = [self.token_pad_id, 0]
        return super(TrainingDatasetRoBERTa, self).paragraph_process(texts, starts, ends, paddings)

    def tfrecord_serialize(self, instances):
        """给原方法补上instance_keys
        """
        instance_keys = ['token_ids', 'mask_ids']
        return super(TrainingDatasetRoBERTa, self).tfrecord_serialize(instances, instance_keys)

    @staticmethod
    def load_tfrecord(record_names, sequence_length, batch_size):
        """给原方法补上parse_function
        """
        def parse_function(serialized):
            features = {
                'token_ids': tf.io.FixedLenFeature([sequence_length], tf.int64),
                'mask_ids': tf.io.FixedLenFeature([sequence_length], tf.int64),
            }
            features = tf.io.parse_single_example(serialized, features)
            token_ids = features['token_ids']
            mask_ids = features['mask_ids']
            segment_ids = K.zeros_like(token_ids, dtype='int64')
            is_masked = K.not_equal(mask_ids, 0)
            masked_token_ids = K.switch(is_masked, mask_ids - 1, token_ids)
            x = {
                'Input-Token': masked_token_ids,
                'Input-Segment': segment_ids,
                'token_ids': token_ids,
                'is_masked': K.cast(is_masked, K.floatx()),
            }
            y = {
                'mlm_loss': K.zeros_like(token_ids[..., 0]),
                'mlm_acc': K.zeros_like(token_ids[..., 0]),
            }
            return x, y

        return TrainingDataset.load_tfrecord(record_names, batch_size, parse_function)




class TextTrainingDataset(TrainingDataset):
    """预训练数据集生成器（RoBERTa模式）
    """
    def __init__(self,
                 tokenizer,
                 word_segment,
                 mask_rate=0.15,
                 sequence_length=512):
        """参数说明：
            tokenizer必须是bert4keras自带的tokenizer类；
            word_segment是任意分词函数。
        """
        super(TrainingDatasetRoBERTa, self).__init__(tokenizer, sequence_length)
        self.word_segment = word_segment
        self.mask_rate = mask_rate

    def token_process(self, token_id):
        """以80%的几率替换为[MASK]，以10%的几率保持不变，
        以10%的几率替换为一个随机token。
        """
        rand = np.random.random()
        if rand <= 0.8:
            return self.token_mask_id
        elif rand <= 0.9:
            return token_id
        else:
            return np.random.randint(0, self.vocab_size)

    def sentence_process(self, text):
        """单个文本的处理函数
        流程：分词，然后转id，按照mask_rate构建全词mask的序列
              来指定哪些token是否要被mask
        """
        words = self.word_segment(text)
        rands = np.random.random(len(words))

        token_ids, mask_ids = [], []
        for rand, word in zip(rands, words):
            word_tokens = self.tokenizer.tokenize(text=word,
                                                  add_cls=False,
                                                  add_sep=False)
            word_token_ids = self.tokenizer.tokens_to_ids(word_tokens)
            token_ids.extend(word_token_ids)

            if rand < self.mask_rate:
                word_mask_ids = [
                    self.token_process(i) + 1 for i in word_token_ids
                ]
            else:
                word_mask_ids = [0] * len(word_tokens)

            mask_ids.extend(word_mask_ids)

        return [token_ids, mask_ids]

    def paragraph_process(self, texts):
        """给原方法补上starts、ends、paddings
        """
        starts = [self.token_cls_id, 0]
        ends = [self.token_sep_id, 0]
        paddings = [self.token_pad_id, 0]
        return super(TrainingDatasetRoBERTa, self).paragraph_process(texts, starts, ends, paddings)

    def tfrecord_serialize(self, instances):
        """给原方法补上instance_keys
        """
        instance_keys = ['token_ids', 'mask_ids']
        return super(TrainingDatasetRoBERTa, self).tfrecord_serialize(instances, instance_keys)

    @staticmethod
    def load_tfrecord(record_names, sequence_length, batch_size):
        """给原方法补上parse_function
        """
        def parse_function(serialized):
            features = {
                'token_ids': tf.io.FixedLenFeature([sequence_length], tf.int64),
                'mask_ids': tf.io.FixedLenFeature([sequence_length], tf.int64),
            }
            features = tf.io.parse_single_example(serialized, features)
            token_ids = features['token_ids']
            mask_ids = features['mask_ids']
            segment_ids = K.zeros_like(token_ids, dtype='int64')
            is_masked = K.not_equal(mask_ids, 0)
            masked_token_ids = K.switch(is_masked, mask_ids - 1, token_ids)
            x = {
                'Input-Token': masked_token_ids,
                'Input-Segment': segment_ids,
                'token_ids': token_ids,
                'is_masked': K.cast(is_masked, K.floatx()),
            }
            y = {
                'mlm_loss': K.zeros_like(token_ids[..., 0]),
                'mlm_acc': K.zeros_like(token_ids[..., 0]),
            }
            return x, y

        return TrainingDataset.load_tfrecord(record_names, batch_size, parse_function)


if __name__ == '__main__':

    from dataset.tokenizer import Tokenizer
    import json, glob, re

    model = 'roberta'
    sequence_length = 512
    workers = 40
    max_queue_size = 4000
    dict_path = '/home/spaces_ac_cn/chinese_L-12_H-768_A-12/vocab.txt'
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    def some_texts():
        filenames = glob.glob('/home/spaces_ac_cn/corpus/*/*/*')
        np.random.shuffle(filenames)
        count, texts = 0, []
        for filename in filenames:
            with open(filename) as f:
                for l in f:
                    l = json.loads(l)['text'].strip()
                    texts.extend(re.findall(u'.*?[\n。]+', l))
                    count += 1
                    if count == 10:  # 10篇文章合在一起再处理
                        yield texts
                        count, texts = 0, []
        if texts:
            yield texts

    assert model in ['roberta', 'gpt']  # 判断是否支持的模型类型

    if model == 'roberta':

        import jieba_fast as jieba
        jieba.initialize()

        def word_segment(text):
            return jieba.lcut(text)

        TD = TrainingDatasetRoBERTa(tokenizer, word_segment, sequence_length=sequence_length)

        for i in range(10):  # 数据重复10遍
            TD.process(
                corpus=tqdm(some_texts()),
                record_name='../corpus_tfrecord/corpus.%s.tfrecord' % i,
                workers=workers,
                max_queue_size=max_queue_size,
            )

    elif model == 'gpt':

        TD = TrainingDatasetGPT(tokenizer, sequence_length=sequence_length)

        TD.process(
            corpus=tqdm(some_texts()),
            record_name='../corpus_tfrecord/corpus.tfrecord',
            workers=workers,
            max_queue_size=max_queue_size,
        )
