import os
import tensorflow as tf
import numpy as np
from model.classification import textcnn,textrnn,textrcnn,textvdcnn,textdcnn,dpcnn,textbirnn
from tensorflow import keras
from config import Config
from dataset.tokenizer import *
from dataset.data_utils import *
import matplotlib.pyplot as plt

# In[16]:


tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


def get_dataset():

    sequence_length = 128

    dict_path = '../conf/vocab.txt'
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    def some_texts():
        filenames = glob.glob('D:\\一键分类\\涉政\\train.txt')
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
        record_name='D:\\一键分类\\涉政\\corpus.tfrecord'
    )
    dataset = TD.load_tfrecord('D:\\一键分类\\涉政\\corpus.tfrecord',sequence_length,32)
    return dataset


def main():
    # fix random seed for reproducibility
    np.random.seed(7)
    # load the dataset but only keep the top n words, zero the rest
    top_words = 10000
    # truncate and pad input sequences
    max_review_length = 80
    (X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=top_words)
    print("X_train", X_train[0:10])
    print("y_train", y_train[0:10])
    print("X_train", type(X_train))
    print("y_train", type(y_train))
    # X_train = tf.convert_to_tensor(X_train)
    # y_train = tf.one_hot(y_train, depth=2)
    print('Pad sequences (samples x time)')
    x_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=max_review_length)
    x_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=max_review_length)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)
    print("x_train", type(x_train))
    print("x_test", type(x_test))
    config = Config(config_file="../conf/train.json")
    batch_size = 1

    print(globals())
    #model = rnn.RNN(config)
    model = textcnn.TextCNN(config)
    check_path = 'ckpt\model.ckpt'
    check_dir = os.path.dirname(check_path)
    latest = tf.train.latest_checkpoint(check_dir)
    print(latest)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(check_path, verbose=1, save_freq=1000)
    checkpoint = tf.train.Checkpoint(model=model)
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    if latest:
        print(latest)
        #model.load_weights(latest)
         # 实例化Checkpoint，指定恢复对象为model
        inputs = keras.layers.Input(
            shape=(config.input_length,),
            name='Input',
        )
        model._set_inputs(inputs)
        model.summary()
        #path = checkpoint.save(check_path)
        checkpoint.restore(tf.train.latest_checkpoint(check_dir)) #.assert_consumed()

        #model.build(input_shape=(None, config.input_length))
        model.save('path_to_saved_model', save_format='tf')


    #model.summary()
    # train
    #dataset = get_dataset()
    history = model.fit(x_train, y_train, batch_size=config.train.batch_size, epochs=1,
              validation_data=(x_test, y_test), verbose=1, callbacks=[cp_callback])

    #model.fit_generator(dataset,steps_per_epoch=1000, epochs=1, verbose=1, callbacks=[cp_callback])

    model.save('path_to_saved_model', save_format='tf')
    path = checkpoint.save(check_path)
    print("model saved to %s" % path)
    '''
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'valiation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'valiation'], loc='upper left')
    plt.show()
    '''
    # evaluate on test set
    scores = model.evaluate(x_test, y_test, batch_size, verbose=1)
    print("Final test loss and accuracy :", scores)
    scores = model.predict(x_test, y_test, batch_size, verbose=1)
    for score in scores:
        print(score)

if __name__ == '__main__':
    main()
