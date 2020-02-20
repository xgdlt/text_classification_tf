# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
from model.regression import mlp,cnn,rnn
from tensorflow import keras
from config import Config
import matplotlib.pyplot as plt
# In[16]:

tf.random.set_seed(22)
np.random.seed(22)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
assert tf.__version__.startswith('2.')
classify = False
def main():
    # fix random seed for reproducibility
    datas = []
    targets = []
    bases = []
    for line in open("train_date_normal_last.csv",encoding="utf-8"):
         lst = line.split("\t")
         if len(lst) == 0 or lst[0] == "指标":
             continue

         data = [float(value) for value in lst[1:-2]]
         datas.append(data)
         targets.append(float(lst[-2]))

    train_test(datas, targets)


def train_test(datas, targets):
    print('datas size:', len(datas))
    print('targets size:', len(targets))
    result = []
    result2 = []
    config = Config(config_file="../conf/train.json")
    print(type(config))
    print(type(config.MLP))
    exit()
    for i in range(1,37):
        x_train = []
        for j in range(i,len(datas)-6):
            data = datas[j:j + 6]
            data.reverse()
            x_train.append(np.array(data))
        x_train = np.array(x_train)
        print("x_train", x_train.shape)
        data = datas[i-1:i+5]
        data.reverse()
        x_test = np.array([np.array(data)])
        print("x_test", x_test.shape)
        '''
        x_train = np.array(datas[i:])
        x_test = np.array(datas[i - 1:i])
       '''
        if not classify:
            y_train = np.array(targets[i:len(datas)-6])

            y_test = np.array(targets[i-1:i])
            print("y_train", y_train.shape)
            print("y_test", y_test.shape)
        else:
            r_train = targets[i:len(datas) - 6]
            c_train = [ 1 if targets[k] > targets[k+1] else 0 for k in range(i,len(datas)-6) ]
            '''
            for i in range(i,len(targets) - 6):
                if targets[i] > targets[i+1]:
                    c_train.append([targets[i],1])
                else:
                    y_train.append([targets[i], 0])
                    '''
            y_train = [np.array(r_train), np.array(c_train)]
            y_test= [np.array([targets[i-1]]),np.array([ 1 if targets[i] > targets[i-1] else 0 ])]

        base_train = np.array(targets[i + 1:len(datas) - 5])
        print("base_train", base_train.shape)

        base_test =  np.array(targets[i:i + 1])
        print("base_test", base_test.shape)
        print("x_train", type(x_train))
        print("x_test", type(x_test))
        '''
        print("x_train", x_train.shape)
        print("y_train", y_train.shape)
        print("x_test", x_test.shape)
        print("y_test", y_test.shape)
        '''

        batch_size = 32
        print(globals())
        model = cnn.model(config)
        #model = rnn.model(config)

        if not classify:
            model.compile(optimizer=keras.optimizers.Adam(0.0005),
                      loss=keras.losses.MeanSquaredError(),
                      metrics=["mse"])
        else:
            model.compile(optimizer=keras.optimizers.Adam(0.0005),
                      loss={"output_1": keras.losses.MeanSquaredError(),
                            "output_2": keras.losses.SparseCategoricalCrossentropy()},
                      metrics={"output_1": "mse", "output_2": "accuracy"})

        if False:
            tf.keras.utils.plot_model(model,to_file='model.png', show_shapes=True, show_layer_names=True)
            exit()
        model.fit([x_train,base_train], y_train, batch_size=config.train.batch_size, epochs=256,
                  validation_data=([x_test,base_test], y_test), verbose=1)

        # evaluate on test set
        scores = model.evaluate([x_test,base_test], y_test, batch_size, verbose=1)
        print("Final test loss and accuracy :", scores)
        predicts = model.predict([x_test,base_test], batch_size=1, verbose=1)
        #predict("predicts", predicts)
        if classify:
            for predict in predicts[0]:
                result.append([base_test[0],y_test[0][0],predict[0]])
            for predict in predicts[1]:
                result2.append([base_test[0], y_test[1][0], np.argwhere(predict == np.max(predict))[0][0]])
        else:
            for predict in predicts:
                result.append([base_test[0], y_test[0], predict[0]])

    print(result)
    print("result:")
    for base_test, y_test, predict in result:
        print(base_test, y_test, predict)
    if classify:
        print("result2:")
        for base_test, y_test, predict in result2:
            print(base_test, y_test, predict)
    return result

if __name__ == '__main__':
    main()
