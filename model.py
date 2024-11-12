import torch.nn as nn

# 定义一个名为CNN的类，它继承自nn.Module，这是构建神经网络的基类
class CNN(nn.Module):
    def __init__(self):
        """
        初始化函数，用于定义网络的层结构
        """
        super(CNN, self).__init__()
        # 定义第一个卷积层，输入通道数为1，输出通道数为32，卷积核大小为5x5
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        # 定义第一个批归一化层，用于对卷积层输出进行归一化，参数32表示通道数
        self.bn1 = nn.BatchNorm2d(32)
        # 定义ReLU激活函数，inplace=True表示直接在原张量上进行操作，节省内存
        self.relu = nn.ReLU(inplace=True)

        # 定义第二个卷积层，输入通道数为32，输出通道数为64，卷积核大小为5x5
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        # 定义第二个批归一化层，用于对第二个卷积层输出进行归一化，参数64表示通道数
        self.bn2 = nn.BatchNorm2d(64)

        # 定义第三个卷积层，输入通道数为64，输出通道数为128，卷积核大小为6x6
        self.conv3 = nn.Conv2d(64, 128, kernel_size=6)
        # 定义第三个批归一化层，用于对第三个卷积层输出进行归一化，参数128表示通道数
        self.bn3 = nn.BatchNorm2d(128)

        # 定义全连接层，将128维的特征映射到10维，通常用于分类任务的最后输出
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        """
        前向传播函数，定义数据在网络中的流动过程
        :param x: 输入数据
        :return: 经过网络处理后的输出数据
        """
        # 依次对输入数据进行卷积、批归一化和ReLU激活操作
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        # 将多维的张量展平为二维张量，方便后续全连接层处理，-1表示自动计算该维度大小
        x = x.view(x.size(0), -1)
        # 经过全连接层处理得到最终输出
        x = self.fc(x)

        return x


# 定义一个名为LSTM的类，同样继承自nn.Module
class LSTM(nn.Module):
    def __init__(self):
        """
        初始化函数，用于定义LSTM网络的层结构
        """
        super(LSTM, self).__init__()
        # 定义LSTM层，input_size表示输入特征维度为196，hidden_size表示隐藏层维度为70，
        # num_layers表示LSTM的层数为1，batch_first=True表示输入数据的第一维是批次大小
        self.lstm = nn.LSTM(input_size=196, hidden_size=70, num_layers=1, batch_first=True)
        # 定义全连接层，将LSTM输出的70维特征映射到10维，用于分类任务的最终输出
        self.fc = nn.Linear(70, 10)

    def forward(self, x):
        """
        前向传播函数，定义数据在LSTM网络中的流动过程
        :param x: 输入数据
        :return: 经过网络处理后的输出数据
        """
        # 对输入数据进行LSTM处理，out是LSTM的输出，(_, _)是隐藏状态等信息，这里暂不使用
        out, (_, _) = self.lstm(x)
        # 取LSTM输出的最后一个时间步的特征，经过全连接层得到最终输出
        out = self.fc(out[:, -1, :])

        return out


if __name__ == "__main__":
    import torch
    # 随机生成一个大小为(64, 1, 196)的张量作为输入数据，这里可以理解为64个样本，
    # 每个样本有1个通道，特征维度为196
    x = torch.rand(size=(64, 1, 196))
    # 创建LSTM网络的实例
    lstm = LSTM()
    # 将输入数据传入LSTM网络进行处理，得到输出
    out = lstm(x)
    # 打印输出数据的形状
    print(out.shape)