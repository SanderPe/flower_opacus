#%%

import pandas as pd
import numpy as np
import torch.nn as nn

from numpy import vstack

from sklearn.preprocessing import  MinMaxScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor

from torch.optim import SGD
from torch.nn import BCELoss

from opacus import PrivacyEngine
from sklearn.metrics import accuracy_score

MAX_GRAD_NORM = 1.2
EPSILON = 50.0
DELTA = 1e-6
EPOCHS = 10

LR = 1e-3
BATCH_SIZE = 512


def accuracy(preds, labels):

    return (preds == labels).mean()


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self):
        # Load dataset
        data = pd.read_csv('../scripts/lightpath_data.csv')
        data = data.loc[:, ~data.columns.isin(['sample','conn_id','src_id','dst_id'])]


        target = pd.read_csv('../scripts/lightpath_target.csv')
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
# prepare the dataset
def prepare_data():
    # load the dataset
    dataset = CSVDataset()
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)
    return train_dl, test_dl

train_dl, test_dl = prepare_data()


# model definition
class neural_network(nn.Module):
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

model = neural_network()
# ----------------------------------------------------
criterion = BCELoss()
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

# enter PrivacyEngine
privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_dl,
    epochs=EPOCHS,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
)
# ----------------------------------------------------
print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")
# train the model
def train_model(model, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.BCELoss()

    losses = []
    top1_acc = []

    for i, (inputs, target) in enumerate(train_loader):
        optimizer.zero_grad()

        # compute output
        output = model(inputs)
        loss = criterion(output, target)

        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()

        # measure accuracy and record loss
        acc = accuracy(preds, labels)

        losses.append(loss.item())
        top1_acc.append(acc)

        loss.backward()
        optimizer.step()
        epsilon = privacy_engine.get_epsilon(DELTA)
        if (i+1) % 200 == 0:
            # epsilon = privacy_engine.get_epsilon(DELTA)
            print(
                f"\tTrain Epoch: {epoch} \t"
                f"Loss: {np.mean(losses):.6f} "
                f"Acc@1: {np.mean(top1_acc) * 100:.6f} "
                f"(ε = {epsilon:.2f}, δ = {DELTA})"
            )
    return round(epsilon,2), round(np.mean(losses),2), round(acc,2)


from tqdm.notebook import tqdm

for epoch in tqdm(range(EPOCHS), desc="Epoch", unit="epoch"):
    epsilon, loss, acc = train_model(model, train_loader, optimizer, epoch + 1)
    print(epoch , epsilon, loss, acc)

# evaluate the model
def evaluate_model(test_dl, model):
    predictions, actuals = list(), list()
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = accuracy_score(actuals, predictions)
    prec = precision_score(actuals, predictions)
    recc = recall_score(actuals, predictions)
    f1 = f1_score(actuals, predictions)

    return acc, prec, recc, f1

acc = evaluate_model(test_dl, model)


# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat



acc, prec, recc, f1 = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
print('Recall: %.3f' % prec)
print('Precision: %.3f' % recc)
print('Precision: %.3f' % f1)
