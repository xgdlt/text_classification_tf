import os
import tensorflow as tf
import numpy as np
from model_tf.classification import textcnn,textrnn,textrcnn
from tensorflow import keras
from config import Config
import matplotlib.pyplot as plt

# In[16]:


tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')


def main():
    # fix random seed for reproducibility
    np.random.seed(7)
    # load the dataset but only keep the top n words, zero the rest
    top_words = 10000
    # truncate and pad input sequences
    max_review_length = 80
    (X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=top_words)
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
    batch_size = 32

    model = textcnn.TextCNN(config)
    #model = textrcnn.TextRCNN(config)

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # train
    history = model.fit(x_train, y_train, batch_size=config.train.batch_size, epochs=config.train.num_epochs,
              validation_data=(x_test, y_test), verbose=1)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['training', 'valiation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['training', 'valiation'], loc='upper left')
    plt.show()

    # evaluate on test set
    scores = model.evaluate(x_test, y_test, batch_size, verbose=1)
    print("Final test loss and accuracy :", scores)


if __name__ == '__main__':
    main()