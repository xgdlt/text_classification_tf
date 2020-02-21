#! -*- coding: utf-8 -*-
import os,sys
sys.path.append("/home/liteng/code/text_classification_tf")
from model.classification import textcnn,textrnn,textrcnn,textvdcnn,textdcnn,dpcnn,textbirnn
from tensorflow import keras
from config import Config
from dataset.tokenizer import *
from dataset.dataset import *

#tf.random.set_seed(22)
#np.random.seed(22)
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


def get_dataset():

    sequence_length = 128

    dict_path = './conf/vocab.txt'
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    def some_texts():
        filenames = glob.glob('./data/train.txt')
        np.random.shuffle(filenames)
        texts = []
        for filename in filenames:
            with open(filename, encoding="utf8") as f:
                for l in f:
                    lst = l.strip().split("\t")
                    if len(lst) < 2:
                        continue
                    texts.append([lst[0], lst[1]])
        return texts[0:1000]

    TD = TrainingDataset(tokenizer, labels=["正常", "涉政负面"], sequence_length=sequence_length)

    TD.process(
        corpus=some_texts(),
        record_name='./data/corpus.tfrecord'
    )
    dataset = TD.load_tfrecord('./data/corpus.tfrecord',sequence_length,32)
    return dataset


def main():

    config = Config(config_file="./conf/train.json")
    batch_size = 1

    print(globals())
    #model = rnn.RNN(config)
    model = textcnn.TextCNN(config)
    check_path = './ckpt/model.ckpt'
    checkpoint = tf.train.Checkpoint(model=model)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(check_path, verbose=1, save_freq=1)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])


    #model.summary()
    # train
    dataset = get_dataset()
    #history = model.fit(x_train, y_train, batch_size=config.train.batch_size, epochs=1,
    #          validation_data=(x_test, y_test), verbose=1, callbacks=[cp_callback])

    model.fit_generator(dataset,steps_per_epoch=10,validation_data=dataset, validation_steps=10,epochs=1, verbose=1, callbacks=[cp_callback])
    model.summary()
    model.save('path_to_saved_model', save_format='tf')
    path = checkpoint.save(check_path)
    print("model saved to %s" % path)

    # evaluate on test set
    scores = model.evaluate_generator(dataset,steps=20,verbose=1)
    print("Final test loss and accuracy :", scores)
    scores = model.predict_generator(dataset,steps=10,verbose=1)
    for score in scores:
        print(score)

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
