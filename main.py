import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Embedding
from keras.models import load_model
from time import time
from keras.callbacks import TensorBoard, ModelCheckpoint


def inputData(path, rasio, kolom):
    data = pd.read_csv(path)
    nDataTrain = math.ceil(data.shape[0]*rasio)
    data_training = data.loc[:nDataTrain, kolom:kolom].values
    data_testing = data.loc[nDataTrain:, kolom:kolom].values
    return data_training, data_testing


def scaleData(data, scaler=None):
    if(scaler != None):
        dataScaled = scaler.transform(data)
        return scaler, dataScaled
    else:
        scaler = StandardScaler()
        scaler.fit(data)
        dataScaled = scaler.transform(data)
        return scaler, dataScaled


def defineModel(dropout_ratio, n_hidden_units, shape, n_lstm_layer=2):
    model = Sequential()
    model.add(LSTM(units=n_hidden_units, return_sequences=True,
                   input_shape=(shape[1], 1)))
    for i in range(n_lstm_layer-2):
        if(i % 2 == 0):
            model.add(LSTM(units=n_hidden_units,
                           return_sequences=True, dropout=dropout_ratio))
        else:
            model.add(LSTM(units=n_hidden_units, return_sequences=True))
    model.add(LSTM(units=n_hidden_units))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    # model.summary()
    return model


def runModel(model, data_x, data_y, n_epoch, batch_size, name):
    name = name+"1"
    tfboard = TensorBoard(log_dir="logs/{}".format(name), histogram_freq=0,
                          batch_size=batch_size, write_graph=True)
    filepath = name+".h5"
    checkPoint = ModelCheckpoint(
        "./models/checkpoints/"+filepath, verbose=1, save_best_only=True, mode="min", monitor="loss")
    model.fit(data_x, data_y, epochs=n_epoch, batch_size=batch_size,
              callbacks=[tfboard, checkPoint])
    model.summary()

    model.save("./models/"+name+".h5")
    return model


def importModel(path):
    return load_model(path)


def predictVal(model, data, scaler):
    prediction = model.predict(data)
    prediction = scaler.inverse_transform(prediction)
    return prediction


def drawPlot(real_val, prediction_val, kolom,title):
    plt.plot(real_val, color='red', label='Real val')
    try:
        plt.plot(prediction_val, color='green', label='predicted val')
    except:
        pass
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel(kolom)
    plt.legend()
    plt.show()


def calculateEval(real_val, prediction_val):
    print('Nilai MSE adalah : ' + str(mean_squared_error(real_val, prediction_val)))
    print('Nilai RMSE adalah : ' +
          str(math.sqrt(mean_squared_error(real_val, prediction_val))))
    print('Nilai MAE adalah : ' +
          str(mean_absolute_error(real_val, prediction_val)))


if __name__ == "__main__":
    path = "./data/kurs_bersih.csv"
    rasio = 0.7
    kolom = "Kurs Jual"

    batch_time = 100
    n_epoch = 100
    batch_size = 32
    dropout_ratio = 0.2
    n_hidden_unit = 100
    n_lstm_layer = 10

    training = False
    # prepare data
    data_training, data_testing = inputData(path, rasio, kolom)
    data_used = np.concatenate((data_training, data_testing), axis=0)
    drawPlot(data_used, None, kolom, "Data Kurs Jual")
    scaler, data_training_scaled = scaleData(data_training)
    scaler, data_testing_scaled = scaleData(data_testing, scaler)
    all_data = np.concatenate(
        (data_training_scaled, data_testing_scaled), axis=0)
    data_test = all_data[len(all_data) -
                         len(data_testing_scaled) - batch_time:]

    # prepare input data training
    train_x = []
    train_y = []
    for i in range(batch_time, len(data_training_scaled)):
        train_x.append(data_training_scaled[i-batch_time:i])
        train_y.append(data_training_scaled[i])
    train_x, train_y = np.array(train_x), np.array(train_y)

    # definisi model
    if training:
        model = defineModel(dropout_ratio, n_hidden_unit,
                            train_x.shape, n_lstm_layer)
        model = runModel(model, train_x, train_y,
                         n_epoch, batch_size, "coba1LSTM")
    else:
        model = importModel("./models/coba1LSTM1.h5")

    # prepare input data testing
    test_x = []
    for i in range(batch_time, len(data_test)):
        test_x.append(data_test[i-batch_time:i])
    test_x = np.array(test_x)

    # predict with data test
    prediction = predictVal(model, train_x, scaler)
    # evaluate with data train
    drawPlot(data_training[batch_time:], prediction, kolom, "Predict with data train")
    calculateEval(data_training[batch_time:], prediction)

    # predict with data test
    prediction = predictVal(model, test_x, scaler)
    # evaluate with data test
    drawPlot(data_testing, prediction, kolom, "Predict with data train")
    calculateEval(data_testing, prediction)
    