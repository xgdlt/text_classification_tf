# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

#from data_util import create_vocabulary,load_data_multilabel
import pickle
import h5py
import os
import shutil
import json
import random
import time
from numba import jit
from knn import fastknn

from util import dataHelper
from util import tokenization

#configuration
FLAGS=tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("device", "0", "which device will use")
tf.app.flags.DEFINE_string("model_name", "textcnn", "which model will use")
tf.app.flags.DEFINE_boolean("new_train",True,"new train or contine train.")
tf.app.flags.DEFINE_boolean("segger",True,"need to segger.")
tf.app.flags.DEFINE_boolean("save_pb",True,"save model pb.")

tf.app.flags.DEFINE_string("model_data_dir","./data","path of training/validation/test data.") #../data/sample_multiple_label.txt
tf.app.flags.DEFINE_string("model_json_path","./model.json","path of vocabulary and label files") #../data/sample_multiple_label.txt

tf.app.flags.DEFINE_float("learning_rate",0.0003,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.65一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir",None,"checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",128,"max sentence length")
tf.app.flags.DEFINE_integer("min_sentence_len",5,"min num for max sentence length")
tf.app.flags.DEFINE_integer("embed_size",128,"embedding size")
tf.app.flags.DEFINE_boolean("is_training_flag",False,"is training.true:tranining,false:testing/deving")
tf.app.flags.DEFINE_boolean("is_deving_flag",True,"is deving.true:deving,false:tranining/testing")
tf.app.flags.DEFINE_boolean("is_testing_flag",True,"is testing.true:testing,false:tranining/deving")

tf.app.flags.DEFINE_integer("num_epochs",1,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",False,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("word2vec_model_path","word2vec-title-desc.bin","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("name_scope","cnn","name scope value.")
tf.app.flags.DEFINE_boolean("multi_label_flag",False,"use multi label or single label.")

tf.app.flags.DEFINE_boolean("use_knn",True,"use knn.")

tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5, "Rate of dropout") #0.5

#textcnn
tf.app.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.app.flags.DEFINE_integer("num_filters", 128, "number of filters") #256--->512
#filter_sizes=list(map(int, FLAGS.filter_sizes.split(",")))

#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    #trainX, trainY, testX, testY = None, None, None, None
    train_example, train_label, dev_example, dev_label, predict_example, predict_label = load_examlpe_data()
    word2index, label2index, trainX, trainY, vaildX, vaildY, testX, testY = cover_example_to_ids(train_example, train_label, dev_example, dev_label, predict_example, predict_label)
    vocab_size = len(word2index)
    print("cnn_model.vocab_size:",vocab_size)
    num_classes=len(label2index)
    print("num_classes:",num_classes)
    #num_examples,FLAGS.sentence_len=trainX.shape
    #print("num_examples of training:",num_examples,";sentence_len:",FLAGS.sentence_len)
    #train, test= load_data_multilabel(FLAGS.traning_data_path,vocabulary_word2index, vocabulary_label2index,FLAGS.sentence_len)
    #trainX, trainY = train;testX, testY = test
    #print some message for debug purpose

    if FLAGS.model_name == "textcnn":
        from textcnn.model import TextCNN as Model
        from textcnn.model import TextCNNConfig as Config
        filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))
        model_config = Config(num_classes,vocab_size,
                              filter_sizes=filter_sizes,
                              num_filters=FLAGS.num_filters,
                              multi_label_flag=FLAGS.multi_label_flag,
                              sequence_length = FLAGS.sentence_len)
        print("model_config =: ", model_config.to_json_string())
        config_file = os.path.join(FLAGS.model_data_dir, "model.config")
        with open(config_file, "w") as fw:
            json.dump(model_config.to_dict(),fw)

    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        model = Model(config=model_config)
        #Initialize Save
        saver=tf.train.Saver(max_to_keep=3)
        if not FLAGS.ckpt_dir:
            FLAGS.ckpt_dir = os.path.join(FLAGS.model_data_dir, "checkpoint/")

        #if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
        if tf.train.get_checkpoint_state(FLAGS.ckpt_dir):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                index2word={v:k for k,v in word2index.items()}
                assign_pretrained_word_embedding(sess, index2word, vocab_size, model,FLAGS.word2vec_model_path)
        curr_epoch, global_step=sess.run([model.epoch_step, model.global_step])
        print("curr_epoch = ", curr_epoch , "global_step = ", global_step)
        #3.feed data & training
        if FLAGS.is_training_flag:
            number_of_training_data=len(trainX)
            batch_size=FLAGS.batch_size
            iteration=0
            for epoch in range(curr_epoch,curr_epoch + FLAGS.num_epochs):
                loss, counter =  0.0, 0
                for start, end in zip(range(0, number_of_training_data+batch_size, batch_size),range(batch_size, number_of_training_data+batch_size, batch_size)):
                    iteration=iteration+1
                    feed_dict = {model.input_x: trainX[start:end],model.dropout_keep_prob: 0.8,model.is_training_flag:FLAGS.is_training_flag}
                    #print("FLAGS.multi_label_flag: ",FLAGS.multi_label_flag)
                    if not FLAGS.multi_label_flag:
                        feed_dict[model.input_y] = trainY[start:end]
                    else:
                        feed_dict[model.input_y_multilabel]=trainY[start:end]
                    curr_loss,lr,_=sess.run([model.loss_val,model.learning_rate,model.train_op],feed_dict)
                    loss,counter=loss+curr_loss,counter+1
                    if counter %50==0:
                        print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tLearning rate:%.5f" %(epoch,counter,loss/float(counter),lr))

                    ########################################################################################################
                    if start%(3000*FLAGS.batch_size)==0: # eval every 3000 steps.
                        if FLAGS.is_deving_flag:
                            eval_accuarcy, eval_loss, f1_score,f1_micro,f1_macro,total_list = do_eval(sess, model, vaildX, vaildY,label2index)
                            print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f" % (epoch, eval_loss, f1_score,f1_micro,f1_macro))
                        # save model to checkpoint
                        save_path = FLAGS.ckpt_dir + "model.ckpt"
                        print("Going to save model..")
                        step = sess.run(model.global_step)
                        print("global_step:", step)
                        saver.save(sess, save_path, global_step=step)
                    ########################################################################################################
                #epoch increment
                print("going to increment epoch counter....")
                sess.run(model.epoch_increment)

                # 4.validation
                print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
                if epoch % FLAGS.validate_every==0:
                    if FLAGS.is_deving_flag:
                        eval_accuarcy, eval_loss,f1_score,f1_micro,f1_macro, total_list=do_eval(sess,model,vaildX,vaildY,label2index)
                        print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f" % (epoch,eval_loss,f1_score,f1_micro,f1_macro))
                    #save model to checkpoint
                    save_path=FLAGS.ckpt_dir+"model.ckpt"
                    step = sess.run(model.global_step)
                    print("global_step:", step)
                    saver.save(sess,save_path,global_step=step)

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        if FLAGS.is_deving_flag:
            dev_ouptput_file = os.path.join(FLAGS.model_data_dir, "dev_result.csv")
            test_accuarcy, test_loss,f1_score,f1_micro,f1_macro,total_list = do_eval(sess, model, vaildX, vaildY,label2index,
                    output_file=dev_ouptput_file,example=dev_example,use_knn=True,train_examples=train_example,train_labels=train_label)
            print("Test accuarcy:%.3f\t Test Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f" % (test_accuarcy, test_loss,f1_score,f1_micro,f1_macro))
        if FLAGS.is_testing_flag:
            ouptput_file =  os.path.join(FLAGS.model_data_dir, "predict_result.csv")
            do_predict(sess, model, testX, testY, label2index,output_file=ouptput_file, example=predict_example)
        # 6.保存pb
        if FLAGS.save_pb:
            output_nodel_names = ["possibility"]
            #output_nodel_names = ["possibility","predictions"]
            output_pb_file =  os.path.join(FLAGS.model_data_dir, "model.pb")
            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,sess.graph_def,output_node_names=output_nodel_names)
            with tf.gfile.FastGFile(output_pb_file,mode="wb") as f:
                f.write(output_graph_def.SerializeToString())
            print("pb file is saved")

    pass


# 在验证集上做验证，报告损失、精确度
def do_eval(sess, model, evalX, evalY, label2index,
            output_file=None,example=None,use_knn=False,train_examples=None,train_labels=None):
    #evalX = evalX[0:3000]
    #evalY = evalY[0:3000]
    num_classes = len(label2index)
    index2label = {}
    for label, index in label2index.items():
        index2label[index] = label
    number_examples = len(evalX)
    eval_accuarcy,eval_loss, eval_counter, eval_f1_score, eval_p, eval_r = 0.0, 0.0, 0, 0.0, 0.0, 0.0
    batch_size = FLAGS.batch_size
    predict = []

    for start, end in zip(range(0, number_examples+batch_size, batch_size), range(batch_size, number_examples+batch_size, batch_size)):
        ''' evaluation in one batch '''
        feed_dict = {model.input_x: evalX[start:end],  model.dropout_keep_prob: 1.0,
                     model.is_training_flag: False}
        if not FLAGS.multi_label_flag:
            feed_dict[model.input_y] = evalY[start:end]
        else:
            feed_dict[model.input_y_multilabel] = evalY[start:end]
        current_eval_loss, possibilitys,accuracy = sess.run(
            [model.loss_val, model.possibility, model.accuracy], feed_dict)
        #print("eval possibility : ", possibility)
        possibilitys = [list(possibility) for possibility in possibilitys]
        predict.extend(possibilitys)
        eval_accuarcy += accuracy
        eval_loss += current_eval_loss
        eval_counter += 1
    y_predict = dataHelper.convert_to_label_num(predict)
    y_true = dataHelper.convert_to_label_num(evalY)
    print("eval predict size : ", len(predict))
    print("eval evalX size : ", len(evalX))
    print("eval evalY size : ", len(evalY))
    #if not FLAGS.multi_label_flag:
    #    predict = [int(ii > 0.5) for ii in predict]


    if use_knn:
        print("beforn knn\n")
        _, _, f1_macro, f1_micro, total_list = fastF1(predict, evalY, num_classes)
        for index, P_R_F1 in enumerate(total_list):
            P, R, F1 = P_R_F1
            print(index2label[index], "P: ", P, "R: ", R, "F1: ", F1)

        knn_model = fastknn.fastknn(train_examples[0:1000], y_true[0:1000])
        print("begin to knn classifier")
        for index, token in enumerate(example):
            start = time.time()
            ratio,train_example,label = knn_model.predict(token)
            #print("cost = %2f"%(start-  time.time()))
            if ratio > 0.9:
                #print("knn: %.2f %s ----- %s   %s"%(ratio,token,train_example, label))
                y_predict[index] = label
        print("after knn\n")

    _, _, f1_macro, f1_micro, total_list = fastF1(predict, evalY, num_classes)
    for index, P_R_F1 in enumerate(total_list):
        P,R,F1 = P_R_F1
        print(index2label[index], "P: ", P, "R: ", R, "F1: ", F1)
    f1_score = (f1_micro+f1_macro)/2.0

    if output_file:
        fw = open(output_file, "w")
        fw.write("total accuarcy: %.4f\n"%(eval_accuarcy/float(eval_counter)))
        fw.write("label\tprecision\taccuracy\tF1\n")
        for index, P_R_F1 in enumerate(total_list):
            P, R, F1 = P_R_F1
            fw.write("%s\t%.4f\t%.4f\t%.4f\n"%(index2label[index],P,R,F1))
        fw.write("\n")
        for index, y_predict_index in enumerate(y_predict):
            fw.write("%s\t%s\t%s\n" % (example[index], index2label.get(y_true[index], None),index2label.get(y_predict_index, None)))
        fw.close()

    return eval_accuarcy/float(eval_counter), eval_loss/float(eval_counter), f1_score, f1_micro, f1_macro,total_list



# 在验证集上做验证，报告损失、精确度
def do_predict(sess, model, predictX, predictY, label2index, output_file=None,example=[]):
    num_classes = len(label2index)
    index2label = {}
    for label, index in label2index.items():
        index2label[index] = label
    number_examples = len(predictX)
    predict_accuarcy,predict_loss, predict_counter, predict_f1_score, predict_p, predict_r = 0.0, 0.0, 0, 0.0, 0.0, 0.0
    batch_size = FLAGS.batch_size
    predict = []

    for start, end in zip(range(0, number_examples+batch_size, batch_size), range(batch_size, number_examples+batch_size, batch_size)):
        ''' predictuation in one batch '''
        feed_dict = {model.input_x: predictX[start:end],  model.dropout_keep_prob: 1.0,
                     model.is_training_flag: False}
        possibilitys = sess.run(model.possibility, feed_dict)
        #print("predict possibility : ", possibility)
        possibilitys = [list(possibility) for possibility in possibilitys]
        predict.extend(possibilitys)
        predict_counter += 1

    print("predict size : ", len(predict))
    print("predictX size : ", len(predictX))
    print("predictY size : ", len(predictY))
    y_predict = dataHelper.convert_to_label_num(predict)
    if output_file:
        fw = open(output_file,"w")
        for index,y_predict_index in enumerate(y_predict):
            fw.write("%s\t%s\n"%(example[index], index2label.get(y_predict_index, None)))
    return y_predict

def readCKPT(sess,saver,checkpoin_path):
    ckpt = tf.train.get_checkpoint_state(checkpoin_path)
    if ckpt:
        print("reading train record from %s"%checkpoin_path)
        saver.restore(sess, checkpoin_path)
        return True
    return False


def savePB(sess,saver,checkpoin_path, output_nodel_names=["possibility"], output_file="model.pb"):
    if readCKPT(sess, saver, checkpoin_path):
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,sess.graph_def, output_node_names=output_nodel_names
        )


def fastF1(predicts,results, num_class):
    ''' f1 score '''
    true_total, r_total, p_total, p, r = 0, 0, 0, 0, 0
    total_list = []
    for trueValue in range(num_class):
        trueNum, recallNum, precisionNum = 0, 0, 0
        for index, values in enumerate(results):
            value = values.index(max(values))
            predcit = predicts[index]
            predcitValue = predcit.index(max(predcit))
            if value == trueValue:
                recallNum += 1
                if value == predcitValue:
                    trueNum += 1
            if predcitValue == trueValue:
                precisionNum += 1
        R = trueNum / recallNum if recallNum else 0
        P = trueNum / precisionNum if precisionNum else 0
        true_total += trueNum
        r_total += recallNum
        p_total += precisionNum
        p += P
        r += R
        f1 = (2 * P * R) / (P + R) if (P + R) else 0
        print(trueValue, P, R, f1)
        total_list.append([P, R, f1])
    p /= num_class
    r /= num_class
    micro_r = true_total / r_total if r_total else 0
    micro_p = true_total / p_total if p_total else 0
    macro_f1 = (2 * p * r) / (p + r) if (p + r) else 0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p +
                                          micro_r) if (micro_p + micro_r) else 0
    print('P: {:.2f}%, R: {:.2f}%, Micro_f1: {:.2f}%, Macro_f1: {:.2f}%'.format(
        p*100, r*100, micro_f1 * 100, macro_f1*100))
    return p, r, macro_f1, micro_f1, total_list

def load_example_and_label(filename, flag=True):
    if flag:
        file = os.path.join(FLAGS.model_data_dir, filename)
        if not os.path.exists(file):
            raise RuntimeError("please download file: " + file)
        example,label = dataHelper.get_data_examples(file, segger=FLAGS.segger)
    else:
        example = []
        label = []
    return example,label


def load_example(filename, flag=True):
    if flag:
        file = os.path.join(FLAGS.model_data_dir, filename)
        if not os.path.exists(file):
            raise RuntimeError("please download file: " + file)
        example = dataHelper.get_predict_data(file, segger=FLAGS.segger)
    else:
        example = []
    return example

def print_example(example,X,label,Y):
    length = 10 if len(example) > 10 else len(example)
    for index in range(length):
        print("example: ", example[index])
        print("example ids: ", X[index])
        print("label: ", label[index])
        print("label ids: ", Y[index])

def get_seq_lentgh(seq_lenth_file, train_example,dev_example,predict_example):
    examples = []
    examples.extend(train_example)
    examples.extend(dev_example)
    examples.extend(predict_example)
    seq_length_cnt = {}
    for example in examples:
        words = example.split(" ")
        seq_length = 0;
        for word in words:
            if word in ["", " "]:
                continue
            seq_length += 1
        seq_length_cnt[seq_length] = seq_length_cnt.get(seq_length,0) + 1
    denominator = sum([v for k, v in seq_length_cnt.items()])
    numerator = 0
    seq_lenths = []
    max_seq_length = -1
    for sequence_length, cnt in sorted(seq_length_cnt.items(), key=lambda x: x[0]):
        numerator += cnt
        ratio = numerator / float(denominator)
        if ratio > 0.95 and max_seq_length < 0:
            max_seq_length = sequence_length
        seq_lenths.append("%d\t%d\t%.4f\n" % (sequence_length, cnt, ratio))
    dataHelper.write_file(seq_lenth_file, seq_lenths)
    return max_seq_length

def load_data(model_data_dir):
    """
    load data from h5py and pickle cache files, which is generate by take step by step of pre-processing.ipynb
    :param model_data_dir:
    :return:
    """
    if not os.path.exists(model_data_dir):
        raise RuntimeError("############################ERROR##############################\n. "
                           "please download file, it include training data and vocabulary & labels. ")
    print("INFO. model_data_dir exists. going to load file")

    train_example, train_label = load_example_and_label( "train.csv", FLAGS.is_training_flag)
    dev_example, dev_label = load_example_and_label("dev.csv", FLAGS.is_deving_flag)
    predict_example,predict_label = load_example_and_label( "test.csv",FLAGS.is_testing_flag)


    label_file = os.path.join(FLAGS.model_data_dir, "label.csv")
    if not os.path.exists(label_file):
        labels = []
        labels.extend(train_label)
        labels.extend(dev_label)
        label_cnt = {}
        for label in labels:
            tags = label.strip().split(" ")
            for tag in tags:
                if tag not in label_cnt:
                    label_cnt[tag] = 0
                label_cnt[tag] += 1
        labels = []
        for label, cnt in label_cnt.items():
            if cnt > 10:
                labels.append(label)
        print("labels : ", label_cnt)
        dataHelper.write_file(label_file,labels)
    label2index =  dataHelper.get_index(label_file)
    print("INFO. label2index: ", label2index)

    word_file = os.path.join(FLAGS.model_data_dir, "word.csv")
    if not os.path.exists(word_file):
        create_word_index(word_file, train_example)
    word2index = dataHelper.get_index(word_file)

    seq_lenth_file = os.path.join(FLAGS.model_data_dir, "seq_length.csv")
    sentence_len = get_seq_lentgh(seq_lenth_file, train_example, dev_example, predict_example)
    if sentence_len < FLAGS.min_sentence_len:
        sentence_len = FLAGS.min_sentence_len
    FLAGS.sentence_len = sentence_len

    print("sentence_len = ",FLAGS.sentence_len )
    train_X = dataHelper.convert_to_ids_by_vocab(FLAGS.sentence_len,word2index, train_example)
    train_Y = dataHelper.convert_to_one_hots(label2index, train_label)
    assert len(train_X) == len(train_Y)
    print("train")
    print_example(train_example,train_X,train_label,train_Y)

    dev_X = dataHelper.convert_to_ids_by_vocab(FLAGS.sentence_len,word2index, dev_example)
    dev_Y = dataHelper.convert_to_one_hots(label2index, dev_label)
    assert len(dev_X) == len(dev_Y)
    print("dev")
    print_example(dev_example, dev_X, dev_label, dev_Y)

    predict_X = dataHelper.convert_to_ids_by_vocab(FLAGS.sentence_len,word2index, predict_example)
    predict_Y = dataHelper.convert_to_one_hots(label2index, predict_label)
    assert len(predict_X) == len(predict_Y)
    print("predict")
    print_example(predict_example, predict_X, predict_label, predict_Y)

    #with open(cache_file_pickle, 'rb') as data_f_pickle:
    #    word2index, label2index=pickle.load(data_f_pickle)
    print("INFO. cache file load successful...")
    return word2index, label2index,train_X,train_Y,dev_X,dev_Y,predict_X, predict_Y,dev_example,predict_example



def load_examlpe_data():
    """
    load data from h5py and pickle cache files, which is generate by take step by step of pre-processing.ipynb
    :return:
    """
    if not os.path.exists(FLAGS.model_data_dir):
        raise RuntimeError("############################ERROR##############################\n. "
                           "please download file, it include training data and vocabulary & labels. ")
    print("INFO. model_data_dir exists. going to load file")

    train_example, train_label = load_example_and_label( "train.csv")
    dev_example, dev_label = load_example_and_label("dev.csv")
    predict_example,predict_label = load_example_and_label( "test.csv")

    return train_example, train_label ,dev_example, dev_label ,predict_example,predict_label


def cover_example_to_ids(train_example, train_label ,dev_example, dev_label ,predict_example,predict_label):

    label_file = os.path.join(FLAGS.model_data_dir, "label.csv")
    if not os.path.exists(label_file):
        labels = []
        labels.extend(train_label)
        labels.extend(dev_label)
        label_cnt = {}
        for label in labels:
            tags = label.strip().split(" ")
            for tag in tags:
                if tag not in label_cnt:
                    label_cnt[tag] = 0
                label_cnt[tag] += 1
        labels = []
        for label, cnt in label_cnt.items():
            if cnt > 10:
                labels.append(label)
        print("labels : ", label_cnt)
        dataHelper.write_file(label_file,labels)
    label2index =  dataHelper.get_index(label_file)
    print("INFO. label2index: ", label2index)

    word_file = os.path.join(FLAGS.model_data_dir, "word.csv")
    if not os.path.exists(word_file):
        create_word_index(word_file, train_example)
    word2index = dataHelper.get_index(word_file)

    seq_lenth_file = os.path.join(FLAGS.model_data_dir, "seq_length.csv")
    sentence_len = get_seq_lentgh(seq_lenth_file, train_example, dev_example, predict_example)
    if sentence_len < FLAGS.min_sentence_len:
        sentence_len = FLAGS.min_sentence_len
    FLAGS.sentence_len = sentence_len
    print("sentence_len = ",FLAGS.sentence_len )
    train_X = dataHelper.convert_to_ids_by_vocab(FLAGS.sentence_len, word2index, train_example)
    train_Y = dataHelper.convert_to_one_hots(label2index, train_label)
    assert len(train_X) == len(train_Y)
    print("train")
    print_example(train_example, train_X, train_label, train_Y)

    dev_X = dataHelper.convert_to_ids_by_vocab(FLAGS.sentence_len,word2index, dev_example)
    dev_Y = dataHelper.convert_to_one_hots(label2index, dev_label)
    assert len(dev_X) == len(dev_Y)
    print("dev")
    print_example(dev_example, dev_X, dev_label, dev_Y)

    predict_X = dataHelper.convert_to_ids_by_vocab(FLAGS.sentence_len,word2index, predict_example)
    predict_Y = dataHelper.convert_to_one_hots(label2index, predict_label)
    assert len(predict_X) == len(predict_Y)
    print("predict")
    print_example(predict_example, predict_X, predict_label, predict_Y)
    #with open(cache_file_pickle, 'rb') as data_f_pickle:
    #    word2index, label2index=pickle.load(data_f_pickle)
    print("INFO. cache file load successful...")
    return word2index, label2index,train_X,train_Y,dev_X,dev_Y,predict_X, predict_Y


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.device
    tf.app.run()
