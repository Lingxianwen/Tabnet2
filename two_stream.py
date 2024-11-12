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

# 设置随机数种子，确保每次运行代码时，随机操作（如数据打乱、初始化等）的结果可复现
seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

# 定义训练的轮数
epochs = 40
# 设置温度参数，用于调整模型输出在计算损失时的缩放比例，可能与某种特定的训练技巧相关（如知识蒸馏等）
temp = 0.3
# 根据是否有可用的CUDA设备，选择使用GPU（cuda）或CPU进行计算
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练集和测试集的CSV文件路径
train_csv_path = r'dataset\UNSW_NB15\UNSW_NB15_training-set.csv'
test_csv_path = r'dataset\UNSW_NB15\UNSW_NB15_testing-set.csv'

# 读取训练集和测试集数据
df_train = pd.read_csv(train_csv_path)
df_test = pd.read_csv(test_csv_path)

# 调用函数删除训练集和测试集的指定列（'id'和'label'），并将处理后的结果保存在df_X中
df_X = drop_extra_label(df_train, df_test, ['id', 'label'])
# 对df_X中的'attack_cat'列进行标签编码，将其转换为数值形式，并将编码后的结果保存在df_Y中，同时从df_X中删除该列
df_Y = LabelEncoder().fit_transform(df_X.pop('attack_cat'))
# 对df_X进行数据预处理（可能包括归一化、编码等操作，具体取决于data_preprocess_unsw函数的实现），并将数据类型转换为np.float32
df_X = data_preprocess_unsw(df_X).astype(np.float32)

# 将数据集划分为训练集、测试集，其中测试集占比为0.25，设置随机种子以确保划分结果可复现
x_train, x_test, y_train, y_test = train_test_split(df_X.values, df_Y, test_size=0.25, random_state=666)
# 进一步将训练集划分为训练集和验证集，其中验证集占训练集的0.25，同样设置随机种子
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=666)

# 创建训练集的数据集对象和数据加载器，设置批量大小为512，并打乱数据顺序
train_ds = UNSWDataset(x_train, y_train)
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)

# 创建测试集的数据集对象和数据加载器，设置批量大小为512，并打乱数据顺序
test_ds = UNSWDataset(x_test, y_test)
test_loader = DataLoader(test_ds, batch_size=512, shuffle=True)

# 创建验证集的数据集对象和数据加载器，设置批量大小为512，并打乱数据顺序
val_ds = UNSWDataset(x_val, y_val)
val_loader = DataLoader(val_ds, batch_size=512, shuffle=True)

# 创建CNN模型实例，并将其移动到指定的设备（GPU或CPU）上
cnn = CNN().to(device)
# 创建LSTM模型实例，并将其移动到指定的设备（GPU或CPU）上
lstm = LSTM().to(device)

# 定义损失函数为交叉熵损失
criterion = nn.CrossEntropyLoss()
# 创建CNN模型的优化器，使用Adam优化算法，设置学习率为1e-3
optim_cnn = torch.optim.Adam(cnn.parameters(), lr=1e-3)
# 创建LSTM模型的优化器，使用Adam优化算法，设置学习率为1e-3
optim_lstm = torch.optim.Adam(lstm.parameters(), lr=1e-3)


def train(cnn, lstm, device, train_loader, epoch, criterion):
    """
    训练函数，用于在给定的数据集上对CNN和LSTM模型进行一轮训练。

    :param cnn: CNN模型实例。
    :param lstm: LSTM模型实例。
    :param device: 计算设备（GPU或CPU）。
    :param train_loader: 训练集数据加载器。
    :param epoch: 当前训练轮数。
    :param criterion: 损失函数。
    """
    # 将模型设置为训练模式，启用梯度计算等训练相关的设置
    cnn.train()
    lstm.train()
    trained_samples = 0
    for batch_idx, (data, target, _) in enumerate(train_loader):
        # 将数据和目标标签移动到指定的计算设备上
        data, target = data.to(device), target.to(device)
        # 对数据进行形状调整，以适应CNN模型的输入要求（这里调整为形状为 (batch_size, 1, 14, 14) 的张量）
        cnn_data = torch.reshape(data, shape=(data.size(0), 1, 14, 14))
        # 对数据进行维度扩展，以适应LSTM模型的输入要求（在维度1上增加一个维度）
        lstm_data = torch.unsqueeze(data, dim=1)

        # 清空CNN和LSTM模型的梯度缓存，为新的一轮梯度计算做准备
        optim_cnn.zero_grad()
        optim_lstm.zero_grad()

        # 通过CNN模型进行前向传播，得到输出
        cnn_output = cnn(cnn_data)
        # 通过LSTM模型进行前向传播，得到输出
        lstm_output = lstm(lstm_data)

        # 计算CNN模型的损失，将输出除以温度参数后再传入损失函数，这里将目标标签转换为长整型
        cnn_loss = criterion(cnn_output/temp, target.long())
        # 计算LSTM模型的损失，将输出除以温度参数后再传入损失函数，这里将目标标签转换为长整型
        lstm_loss = criterion(lstm_output/temp, target.long())
        # 总损失为CNN模型损失和LSTM模型损失之和
        loss = cnn_loss + lstm_loss

        # 进行反向传播，计算梯度
        loss.backward()

        # 根据计算得到的梯度，更新CNN模型的参数
        optim_cnn.step()
        # 根据计算得到的梯度，更新LSTM模型的参数
        optim_lstm.step()

        # 统计已训练的样本数量
        trained_samples += len(data)
        # 计算训练进度，以百分比形式表示，并打印训练进度信息
        progress = math.ceil(batch_idx / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
              (epoch, trained_samples, len(train_loader.dataset),
               '-' * progress + '>', progress * 2), end='')


def test(cnn, lstm, device, loader, criterion):
    """
    测试函数，用于在给定的数据集上对CNN和LSTM模型进行测试，并计算相关的评估指标。

    :param cnn: CNN模型实例。
    :param lstm: LSTM模型实例。
    :param device: 计算设备（GPU或CPU）。
    :param loader: 测试集或验证集的数据加载器。
    :param criterion: 损失函数。
    :return: 准确率、精确率、召回率和F1值。
    """
    # 将模型设置为评估模式，关闭梯度计算等训练相关的设置
    cnn.eval()
    lstm.eval()
    correct = 0
    test_loss = 0.0
    # 创建用于存储所有目标标签的张量，初始化为全零，数据类型为torch.int32，并移动到指定设备上
    total_target = torch.zeros(len(loader.dataset), dtype=torch.int32).to(device)
    # 创建用于存储所有预测结果的张量，初始化为全零，数据类型为torch.int64，并移动到指定设备上
    total_pred = torch.zeros(len(loader.dataset), dtype=torch.int64).to(device)
    with torch.no_grad():
        for batch_idx, (data, target, input_indices) in enumerate(loader):
            # 将数据和目标标签移动到指定的计算设备上
            data, target = data.to(device), target.to(device)
            # 对数据进行形状调整，以适应CNN模型的输入要求（这里调整为形状为 (batch_size, 1, 14, 14) 的张量）
            cnn_data = torch.reshape(data, shape=(data.size(0), 1, 14, 14))
            # 对数据进行维度扩展，以适应LSTM模型的输入要求（在维度1上增加一个维度）
            lstm_data = torch.unsqueeze(data, dim=1)

            # 将目标标签存储到total_target张量中对应的位置
            total_target[input_indices] = target
            # 通过CNN模型进行前向传播，得到输出
            cnn_output = cnn(cnn_data)
            # 通过LSTM模型进行前向传播，得到输出
            lstm_output = lstm(lstm_data)

            # 计算测试损失，将CNN和LSTM模型的输出相加后传入损失函数，这里将目标标签转换为长整型，并累加损失值
            test_loss += criterion(cnn_output+lstm_output, target.long()).item()
            # 根据CNN和LSTM模型的输出平均值，预测最可能的类别，并将预测结果存储到total_pred张量中对应的位置
            predicted = ((cnn_output+lstm_output)/2).argmax(dim=1)
            total_pred[input_indices] = predicted
            # 统计预测正确的样本数量
            correct += predicted.eq(target).sum().item()

    # 计算精确率，根据预测结果和目标标签计算，采用加权平均方式
    pre_score = precision_score(total_target.cpu().numpy(), total_pred.cpu().numpy(), average='weighted')
    # 计算召回率，根据预测结果和目标标签计算，采用加权平均方式
    re_score = recall_score(total_target.cpu().numpy(), total_pred.cpu().numpy(), average='weighted')
    # 计算F1值，根据预测结果和目标标签计算，采用加权平均方式
    f_score = f1_score(total_target.cpu().numpy(), total_pred.cpu().numpy(), average='weighted')

    print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(
        test_loss/len(loader.dataset), correct, len(loader.dataset),
        100. * correct / len(loader.dataset)))

    return correct / len(loader.dataset), pre_score, re_score, f_score


# 进行多轮训练和测试
for epoch in range(1, epochs+1):
    train(cnn, lstm, device, train_loader, epoch, criterion)
    acc, _, _, _ = test(cnn, lstm, device, val_loader, criterion)

acc, precision, recall, f1 = test(cnn, lstm, device, test_loader, criterion)
print("acc is {0}, precision is {1}, recall is {2}, f1 score is {3}".format(acc, precision, recall, f1))