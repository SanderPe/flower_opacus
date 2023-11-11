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
from torch import no_grad
from torch.nn import BCELoss, CrossEntropyLoss
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import SGD
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import OrderedDict

import torch

EPOCHS = 20
LR = 1e-3
BATCH_SIZE = 512

def accuracy(preds, labels):

    return (preds == labels).mean()


# dataset definition
class CSVDataset():
    # load the dataset
    def __init__(self, number):
        # Load dataset
        data = pd.read_csv('../scripts/lightpath_community_' + str(number) + '_data.csv')
        data = data.loc[:, ~data.columns.isin(['sample','conn_id','src_id','dst_id'])]

        target = pd.read_csv('../scripts/lightpath_community_' + str(number) + '_target.csv')
        target = target['class']
        data['target'] = target
        data = data.sample(100000)
        target = data['target']
        data = data.loc[:, data.columns!='target']
        target = np.array(target).reshape(len(target),1)
        data = data.values

        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        data = scaler.transform(data)


        self.X = data
        self.y = target
        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(32, 64)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(64, 256)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(256, 1)
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x

def prepare_data(number):




    # load the dataset
    dataset = CSVDataset(number)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)
    num_examples = {"trainset": len(train), "testset": len(test)}
    return train_dl, test_dl, num_examples





def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = BCELoss()
    optimizer = SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()

    losses = []
    top1_acc = []
    for _ in range(epochs):
        for i, (inputs, target) in enumerate(trainloader):
            optimizer.zero_grad()

            # compute output
            output = net(inputs)
            loss = criterion(output, target)

            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            losses.append(loss.item())

            loss.backward()
            optimizer.step()

def test(net, test_dl):
    """Validate the network on the entire test set."""

    criterion = BCELoss()
    correct, total, loss = 0, 0, 0.0

    predictions, actuals = list(), list()

    with no_grad():
        for i, (inputs, labels) in enumerate(test_dl):

            yhat = net(inputs)


            loss += criterion(yhat, labels).item()


            # evaluate the model on the test set

            # retrieve numpy array
            yhat = yhat.detach().numpy()
            actual = labels.numpy()
            actual = actual.reshape((len(actual), 1))
            # round to class values
            yhat = yhat.round()

            # store
            predictions.append(yhat)
            actuals.append(actual)
    predictions, actuals = np.vstack(predictions), np.vstack(actuals)
    # calculate accuracy

    acc = accuracy_score(actuals, predictions)
    prec = precision_score(actuals, predictions)
    recc = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)

    return loss, acc, prec, recc, f1


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, number):
        # Load model and data
        self.net = Net()
        self.train_dl, self.test_dl, self.num_examples = prepare_data(number)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.train_dl, epochs=EPOCHS)
        return self.get_parameters(config={}), self.num_examples["trainset"], {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, prec, recc, f1 = test(self.net, self.test_dl)
        return float(loss), self.num_examples["testset"], {"accuracy": float(accuracy),"precision": float(prec),"recall": float(recc),"f1": float(f1)}







