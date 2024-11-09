import torch
import math
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import recall_score, precision_score, f1_score

from utils import drop_extra_label, data_preprocess_unsw
from data_loader import UNSWDataset
from model import CNN, LSTM


# 设置随机数种子
seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

epochs = 40
temp = 0.3    # set up temperature
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_csv_path = r'dataset\UNSW_NB15\UNSW_NB15_training-set.csv'
test_csv_path = r'dataset\UNSW_NB15\UNSW_NB15_testing-set.csv'

df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path)
df_X = drop_extra_label(df_train, df_test, ['id', 'label'])
df_Y = LabelEncoder().fit_transform(df_X.pop('attack_cat'))
df_X = data_preprocess_unsw(df_X).astype(np.float32)

x_train, x_test, y_train, y_test = train_test_split(df_X.values, df_Y, test_size=0.25, random_state=666)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=666)

train_ds = UNSWDataset(x_train, y_train)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

test_ds = UNSWDataset(x_test, y_test)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=True)

val_ds = UNSWDataset(x_val, y_val)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=True)

cnn = CNN().to(device)
lstm = LSTM().to(device)

criterion = nn.CrossEntropyLoss()
optim_cnn = torch.optim.Adam(cnn.parameters(), lr=1e-3)
optim_lstm = torch.optim.Adam(lstm.parameters(), lr=1e-3)


def train(cnn, lstm, device, train_loader, epoch, criterion):
    cnn.train()
    lstm.train()
    trained_samples = 0
    for batch_idx, (data, target, _) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        cnn_data = torch.reshape(data, shape=(data.size(0), 1, 14, 14))
        lstm_data = torch.unsqueeze(data, dim=1)

        optim_cnn.zero_grad()
        optim_lstm.zero_grad()

        cnn_output = cnn(cnn_data)
        lstm_output = lstm(lstm_data)

        cnn_loss = criterion(cnn_output/temp, target.long())
        lstm_loss = criterion(lstm_output/temp, target.long())
        loss = cnn_loss + lstm_loss

        loss.backward()

        optim_cnn.step()
        optim_lstm.step()

        trained_samples += len(data)
        progress = math.ceil(batch_idx / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
              (epoch, trained_samples, len(train_loader.dataset),
               '-' * progress + '>', progress * 2), end='')


def test(cnn, lstm, device, loader, criterion):
    cnn.eval()
    lstm.eval()
    correct = 0
    test_loss = 0.0
    total_target = torch.zeros(len(loader.dataset), dtype=torch.int32).to(device)
    total_pred = torch.zeros(len(loader.dataset), dtype=torch.int64).to(device)
    with torch.no_grad():
        for batch_idx, (data, target, input_indices) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            cnn_data = torch.reshape(data, shape=(data.size(0), 1, 14, 14))
            lstm_data = torch.unsqueeze(data, dim=1)

            total_target[input_indices] = target
            # model output
            cnn_output = cnn(cnn_data)
            lstm_output = lstm(lstm_data)

            test_loss += criterion(cnn_output+lstm_output, target.long()).item()
            predicted = ((cnn_output+lstm_output)/2).argmax(dim=1)
            total_pred[input_indices] = predicted
            correct += predicted.eq(target).sum().item()

    pre_score = precision_score(total_target.cpu().numpy(), total_pred.cpu().numpy(), average='weighted')
    re_score = recall_score(total_target.cpu().numpy(), total_pred.cpu().numpy(), average='weighted')
    f_score = f1_score(total_target.cpu().numpy(), total_pred.cpu().numpy(), average='weighted')

    print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(
        test_loss/len(loader.dataset), correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))

    return correct / len(loader.dataset), pre_score, re_score, f_score


for epoch in range(1, epochs+1):
    train(cnn, lstm, device, train_loader, epoch, criterion)
    acc, _, _, _ = test(cnn, lstm, device, val_loader, criterion)

acc, precision, recall, f1 = test(cnn, lstm, device, test_loader, criterion)
print("acc is {0}, precision is {1}, recall is {2}, f1 score is {3}".format(acc, precision, recall, f1))