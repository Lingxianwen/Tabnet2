import torch
import numpy as np

from torch.utils.data import Dataset

# 这里定义了数据集中不同类型的标签对应关系，可能用于后续的数据处理或模型训练中的分类任务
# 0: Dos  1: Normal  2: Probe  3: R2L  4: U2R
# 0: BENIGN 1: Dos GoldenEye 2: DoS Hulk 3: DoS Slowhttptest 4: DoS slowloris 5: Heartbleed

# 定义NSL_KDD数据集类，继承自Dataset类，用于处理NSL_KDD数据集相关操作
class NSL_KDDDataset(Dataset):
    def __init__(self, train_data, test_data, image_size=8):
        """
        初始化函数，用于设置NSL_KDD数据集的相关参数和数据。

        :param train_data: 训练数据，通常是一个二维数组或类似的数据结构，包含了训练样本的特征信息。
        :param test_data: 测试数据，同样通常是一个二维数组或类似的数据结构，包含了测试样本的特征信息。
        :param image_size: 图像尺寸，默认为8，用于将数据重塑为特定形状的图像数据（可能在后续处理中与图像相关的操作有关）。
        """
        super(NSL_KDDDataset, self).__init__()

        self.image_size = image_size
        self.train_data = train_data
        self.test_data = test_data
        # 计算需要填充的大小，使得训练数据的长度达到64（可能是为了满足某种模型输入的固定尺寸要求）
        self.pad_size = 64 - self.train_data.shape[1]

    def __len__(self):
        """
        返回测试数据的长度，即数据集中测试样本的数量。

        :return: 测试数据的长度。
        """
        return len(self.test_data)

    def __getitem__(self, index):
        """
        根据给定的索引获取数据集中的一个样本及其对应的标签和索引信息。

        :param index: 要获取的样本在测试数据中的索引。
        :return: 一个包含图像数据（经过处理和转换后的训练数据）、标签和索引的元组。
        """
        label = self.test_data[index]
        # 对训练数据中的指定样本进行填充操作，在末尾填充0，使其长度达到64
        img = np.pad(self.train_data[index], (0, self.pad_size), 'constant')
        # 将填充后的一维数据重塑为三维图像数据的形状，通道数为1，高度和宽度为image_size
        img = np.reshape(img, (1, self.image_size, self.image_size))
        # 将numpy数组转换为torch张量，以便在PyTorch中进行后续处理
        img = torch.from_numpy(img)

        return img, label, index


# 定义UNSW数据集类，继承自Dataset类，用于处理UNSW数据集相关操作
class UNSWDataset(Dataset):
    def __init__(self, train_data, test_data):
        """
        初始化函数，用于设置UNSW数据集的相关参数和数据。

        :param train_data: 训练数据，通常是一个二维数组或类似的数据结构，包含了训练样本的特征信息。
        :param test_data: 测试数据，同样通常是一个二维数组或类似的数据结构，包含了测试样本的特征信息。
        """
        super(UNSWDataset, self).__init__()

        self.train_data = train_data
        self.test_data = test_data

    def __len__(self):
        """
        返回测试数据的长度，即数据集中测试样本的数量。

        :return: 测试数据的长度。
        """
        return len(self.test_data)

    def __getitem__(self, index):
        """
        根据给定的索引获取数据集中的一个样本及其对应的标签和索引信息。

        :param index: 要获取的样本在测试数据中的索引。
        :return: 一个包含图像数据（经过转换后的训练数据）、标签和索引的元组。
        """
        label = self.test_data[index]
        # 将训练数据中的指定样本从numpy数组转换为torch张量，以便在PyTorch中进行后续处理
        img = torch.from_numpy(self.train_data[index])

        return img, label, index


if __name__ == "__main__":
    from utils import drop_extra_label
    import pandas as pd
    from sklearn import preprocessing
    train_csv_path = r'dataset\UNSW_NB15/UNSW_NB15_training-set.csv'
    test_csv_path = r'dataset\UNSW_NB15/UNSW_NB15_testing-set.csv'
    # 调用drop_extra_label函数，从读取的训练集和测试集数据中删除指定的列（'id'和'label'），并返回处理后的数据
    data = drop_extra_label(pd.read_csv(train_csv_path), pd.read_csv(test_csv_path), ['id', 'label'])
    Y = data.pop('attack_cat').values
    # 使用LabelEncoder对弹出的'attack_cat'列的值进行标签编码，将其转换为数值形式，以便后续处理
    Y = preprocessing.LabelEncoder().fit_transform(Y)
    from collections import Counter
    # 打印经过标签编码后的标签值的计数情况，可用于查看各类别的分布情况
    print(Counter(Y))