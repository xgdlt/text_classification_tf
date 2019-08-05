# -*- coding: utf-8 -*-
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8') #gb2312
#training the model.
#process--->1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
#import sys
#reload(sys)
#sys.setdefaultencoding('utf8')
import tensorflow as tf
import numpy as np

#from data_util import create_vocabulary,load_data_multilabel
import pickle
import h5py
import os
import random
from numba import jit

from util import dataHelper

#configuration
FLAGS=tf.app.flags.FLAGS

#tf.app.flags.DEFINE_string("traning_data_path","../data/sample_multiple_label.txt","path of traning data.") #../data/sample_multiple_label.txt
#tf.app.flags.DEFINE_integer("vocab_size",100000,"maximum vocab size.")

tf.app.flags.DEFINE_string("model_name", "textcnn", "which model will use")

tf.app.flags.DEFINE_string("model_data_dir","./data","path of training/validation/test data.") #../data/sample_multiple_label.txt
tf.app.flags.DEFINE_string("model_json_path","./model.json","path of vocabulary and label files") #../data/sample_multiple_label.txt

tf.app.flags.DEFINE_float("learning_rate",0.0003,"learning rate")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.") #批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.") #6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.") #0.65一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir","checkpoint/","checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len",128,"max sentence length")
tf.app.flags.DEFINE_integer("embed_size",128,"embedding size")
tf.app.flags.DEFINE_boolean("is_training_flag",True,"is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs",10,"number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.") #每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding",False,"whether to use embedding or not.")
tf.app.flags.DEFINE_string("word2vec_model_path","word2vec-title-desc.bin","word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("name_scope","cnn","name scope value.")
tf.app.flags.DEFINE_boolean("multi_label_flag",False,"use multi label or single label.")

#textcnn
tf.app.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.app.flags.DEFINE_integer("num_filters", 128, "number of filters") #256--->512
#filter_sizes=list(map(int, FLAGS.filter_sizes.split(",")))





#1.load data(X:list of lint,y:int). 2.create session. 3.feed data. 4.training (5.validation) ,(6.prediction)
def main(_):
    #trainX, trainY, testX, testY = None, None, None, None
    #vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, _= create_vocabulary(FLAGS.traning_data_path,FLAGS.vocab_size,name_scope=FLAGS.name_scope)
    word2index, label2index, trainX, trainY, vaildX, vaildY, testX, testY=load_data(FLAGS.model_data_dir)
    vocab_size = len(word2index)
    print("cnn_model.vocab_size:",vocab_size)
    num_classes=len(label2index)
    print("num_classes:",num_classes)
    #num_examples,FLAGS.sentence_len=trainX.shape
    #print("num_examples of training:",num_examples,";sentence_len:",FLAGS.sentence_len)
    #train, test= load_data_multilabel(FLAGS.traning_data_path,vocabulary_word2index, vocabulary_label2index,FLAGS.sentence_len)
    #trainX, trainY = train;testX, testY = test
    #print some message for debug purpose
    print("trainX[0:10]:", trainX[0:10])
    print("trainY[0]:", trainY[0:10])
    print("train_y_short:", trainY[0])

    if FLAGS.model_name == "textcnn":
        from textcnn.model import TextCNN as Model
        from textcnn.model import TextCNNConfig as Config
        filter_sizes = list(map(int, FLAGS.filter_sizes.split(",")))
        model_config = Config(num_classes,vocab_size,
                              filter_sizes=filter_sizes,
                              num_filters=FLAGS.num_filters,
                              multi_label_flag=FLAGS.multi_label_flag)
        print("model_config =: ", model_config.to_json_string())

    #2.create session.
    config=tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        #Instantiate Model
        model = Model(config=model_config)
        #Initialize Save
        saver=tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint.")
            saver.restore(sess,tf.train.latest_checkpoint(FLAGS.ckpt_dir))
            #for i in range(3): #decay learning rate if necessary.
            #    print(i,"Going to decay learning rate by half.")
            #    sess.run(textCNN.learning_rate_decay_half_op)
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding: #load pre-trained word embedding
                index2word={v:k for k,v in word2index.items()}
                assign_pretrained_word_embedding(sess, index2word, vocab_size, model,FLAGS.word2vec_model_path)
        curr_epoch, global_step=sess.run(model.epoch_step, model.global_step)
        #3.feed data & training
        number_of_training_data=len(trainX)
        batch_size=FLAGS.batch_size
        iteration=0
        for epoch in range(curr_epoch,FLAGS.num_epochs):
            loss, counter =  0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size),range(batch_size, number_of_training_data, batch_size)):
                iteration=iteration+1
                if epoch==0 and counter==0:
                    print("trainX[start:end]:",trainX[start:end])
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
                    eval_loss, f1_score,f1_micro,f1_macro = do_eval(sess, model, vaildX, vaildY,num_classes)
                    print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f" % (epoch, eval_loss, f1_score,f1_micro,f1_macro))
                    # save model to checkpoint
                    save_path = FLAGS.ckpt_dir + "model.ckpt"
                    print("Going to save model..")
                    saver.save(sess, save_path, global_step=epoch)
                ########################################################################################################
            #epoch increment
            print("going to increment epoch counter....")
            sess.run(model.epoch_increment)

            # 4.validation
            print(epoch,FLAGS.validate_every,(epoch % FLAGS.validate_every==0))
            if epoch % FLAGS.validate_every==0:
                eval_loss,f1_score,f1_micro,f1_macro=do_eval(sess,model,testX,testY,num_classes)
                print("Epoch %d Validation Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f" % (epoch,eval_loss,f1_score,f1_micro,f1_macro))
                #save model to checkpoint
                save_path=FLAGS.ckpt_dir+"model.ckpt"
                saver.save(sess,save_path,global_step=epoch)

        # 5.最后在测试集上做测试，并报告测试准确率 Test
        test_loss,f1_score,f1_micro,f1_macro = do_eval(sess, model, testX, testY,num_classes)
        print("Test Loss:%.3f\tF1 Score:%.3f\tF1_micro:%.3f\tF1_macro:%.3f" % ( test_loss,f1_score,f1_micro,f1_macro))
    pass


# 在验证集上做验证，报告损失、精确度
def do_eval(sess, model, evalX, evalY, num_classes):
    evalX = evalX[0:3000]
    evalY = evalY[0:3000]
    number_examples = len(evalX)
    eval_loss, eval_counter, eval_f1_score, eval_p, eval_r = 0.0, 0, 0.0, 0.0, 0.0
    batch_size = FLAGS.batch_size
    predict = []

    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        ''' evaluation in one batch '''
        feed_dict = {model.input_x: evalX[start:end],  model.dropout_keep_prob: 1.0,
                     model.is_training_flag: False}
        if not FLAGS.multi_label_flag:
            feed_dict[model.input_y] = evalY[start:end]
        else:
            feed_dict[model.input_y_multilabel] = evalY[start:end]
        current_eval_loss, logits = sess.run(
            [model.loss_val, model.logits], feed_dict)

        predict.extend(logits[0])
        eval_loss += current_eval_loss
        eval_counter += 1

    if not FLAGS.multi_label_flag:
        predict = [int(ii > 0.5) for ii in predict]
    _, _, f1_macro, f1_micro, _ = fastF1(predict, evalY)
    f1_score = (f1_micro+f1_macro)/2.0
    return eval_loss/float(eval_counter), f1_score, f1_micro, f1_macro

def fastF1(result, predict):
    ''' f1 score '''
    true_total, r_total, p_total, p, r = 0, 0, 0, 0, 0
    total_list = []
    for trueValue in range(6):
        trueNum, recallNum, precisionNum = 0, 0, 0
        for index, values in enumerate(result):
            if values == trueValue:
                recallNum += 1
                if values == predict[index]:
                    trueNum += 1
            if predict[index] == trueValue:
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
    p /= 6
    r /= 6
    micro_r = true_total / r_total if r_total else 0
    micro_p = true_total / p_total if p_total else 0
    macro_f1 = (2 * p * r) / (p + r) if (p + r) else 0
    micro_f1 = (2 * micro_p * micro_r) / (micro_p +
                                          micro_r) if (micro_p + micro_r) else 0
    print('P: {:.2f}%, R: {:.2f}%, Micro_f1: {:.2f}%, Macro_f1: {:.2f}%'.format(
        p*100, r*100, micro_f1 * 100, macro_f1*100))
    return p, r, macro_f1, micro_f1, total_list

def assign_pretrained_word_embedding(sess,vocabulary_index2word,vocab_size,model,word2vec_model_path):
    import word2vec # we put import here so that many people who do not use word2vec do not need to install this package. you can move import to the beginning of this file.
    print("using pre-trained word emebedding.started.word2vec_model_path:",word2vec_model_path)
    word2vec_model = word2vec.load(word2vec_model_path, kind='bin')
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # create an empty word_embedding list.
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0;
    count_not_exist = 0
    for i in range(2, vocab_size):  # loop each word. notice that the first two words are pad and unknown token
        word = vocabulary_index2word[i]  # get a word
        embedding = None
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding;
            count_exist = count_exist + 1  # assign array to this word.
        else:  # no embedding for this word
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size);
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.constant(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(model.Embedding,word_embedding)  # assign this value to our embedding variables of our model.
    sess.run(t_assign_embedding);
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding.ended...")

def process_data(model_data_dir):
    pass

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



    train_file = os.path.join(FLAGS.model_data_dir, "train.csv")
    train_example,train_label = dataHelper.get_data_examples(train_file)

    print("train example: ")
    print(train_example[0:10])
    print(train_label[0:10])


    dev_file = os.path.join(FLAGS.model_data_dir, "dev.csv")
    dev_example,dev_label = dataHelper.get_data_examples(dev_file)

    print("dev example: ")
    print(dev_example[0:10])
    print(dev_label[0:10])

    predict_file = os.path.join(FLAGS.model_data_dir, "test.csv")
    predict_example, predict_label = dataHelper.get_data_examples(predict_file)
    print("predict example: ")
    print(predict_example[0:10])
    print(predict_label[0:10])


    label_file = os.path.join(FLAGS.model_data_dir, "label.csv")
    if not os.path.exists(label_file):
        labels = []
        labels.extend(train_label)
        labels.extend(dev_label)
        labels.extend(predict_label)
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
        examples= []
        examples.extend(train_example)
        examples.extend(dev_example)
        examples.extend(predict_example)
        words_cnt = {}
        for example in examples:
            words = example.split(" ")
            for word in words:
                if word not in words_cnt:
                    words_cnt[word] = 0
                words_cnt[word] += 1
        words = []
        words.append("[PAD]")
        words.append("[UNK]")
        for word, cnt in words_cnt.items():
            if cnt > 2:
                words.append(word)
        dataHelper.write_file(word_file, words)
    word2index = dataHelper.get_index(word_file)


    train_X = dataHelper.convert_to_ids_by_vocab(FLAGS.sentence_len,word2index, train_example)
    train_Y = dataHelper.convert_to_one_hots(label2index, train_label)

    dev_X = dataHelper.convert_to_ids_by_vocab(FLAGS.sentence_len,word2index, dev_example)
    dev_Y = dataHelper.convert_to_one_hots(label2index, dev_label)

    predict_X = dataHelper.convert_to_ids_by_vocab(FLAGS.sentence_len,word2index, predict_example)
    predict_Y = dataHelper.convert_to_one_hots(label2index, predict_label)

    #with open(cache_file_pickle, 'rb') as data_f_pickle:
    #    word2index, label2index=pickle.load(data_f_pickle)
    print("INFO. cache file load successful...")
    return word2index, label2index,train_X,train_Y,dev_X,dev_Y,predict_X, predict_Y


if __name__ == "__main__":
    tf.app.run()
