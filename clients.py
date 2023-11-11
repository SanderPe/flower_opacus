import flwr as fl
import tensorflow as tf
from tensorflow import keras
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

class FlowerClient(fl.client.NumPyClient):

    def __init__(self, number):

        self.data = pd.read_csv('../../scripts/lightpath_community_' + str(number) + '_data.csv')

        self.data = self.data.loc[:, ~self.data.columns.isin(['sample', 'conn_id', 'src_id', 'dst_id'])]
        self.target = pd.read_csv('../../scripts/lightpath_community_' + str(number) + '_target.csv')

        self.target = self.target['class']

        self.data['target'] = self.target
        self.data = self.data.sample(20000)

        self.target = self.data['target']
        self.data = self.data.loc[:, self.data.columns != 'target']

        self.target = np.array(self.target).reshape(len(self.target), 1)
        self.data = self.data.values

        self.model = keras.Sequential([
            keras.layers.Flatten(input_shape=(self.data.shape[1],)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy","Precision","Recall"])

        self.scaler = MinMaxScaler()
        self.scaler = self.scaler.fit(self.data)
        self.data = self.scaler.transform(self.data)

        # split data into training and test set
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.data, self.target,
                                                                                test_size=0.2, random_state=42)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=5)
        r = self.model.fit(self.x_train, self.y_train, epochs=10, validation_split=0.3, verbose=0, callbacks=[es])

        hist = r.history
        print("Fit history : ", hist)
        # list = hist['val_accuracy']{'val_acc' : val_acc}
        # val_acc = list[-1]
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy, precision, recall = self.model.evaluate(self.x_test, self.y_test, verbose=0)

        return loss, len(self.x_test), {"accuracy": accuracy, "recall" : recall, "precision" : precision}



