import torch
import numpy as np

from torch.utils.data import Dataset

# 0: Dos  1: Normal  2: Probe  3: R2L  4: U2R
# 0: BENIGN 1: Dos GoldenEye 2: DoS Hulk 3: DoS Slowhttptest 4: DoS slowloris 5: Heartbleed


class NSL_KDDDataset(Dataset):
    def __init__(self, train_data, test_data, image_size=8):
        super(NSL_KDDDataset, self).__init__()

        self.image_size = image_size
        self.train_data = train_data
        self.test_data = test_data
        self.pad_size = 64 - self.train_data.shape[1]

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        label = self.test_data[index]
        img = np.pad(self.train_data[index], (0, self.pad_size), 'constant')
        img = np.reshape(img, (1, self.image_size, self.image_size))
        img = torch.from_numpy(img)

        return img, label, index


class UNSWDataset(Dataset):
    def __init__(self, train_data, test_data):
        super(UNSWDataset, self).__init__()

        self.train_data = train_data
        self.test_data = test_data

    def __len__(self):
        return len(self.test_data)

    def __getitem__(self, index):
        label = self.test_data[index]
        img = torch.from_numpy(self.train_data[index])

        return img, label, index


if __name__ == "__main__":
    from utils import drop_extra_label
    import pandas as pd
    from sklearn import preprocessing
    train_csv_path = r'dataset\UNSW_NB15/UNSW_NB15_training-set.csv'
    test_csv_path = r'dataset\UNSW_NB15/UNSW_NB15_testing-set.csv'
    data = drop_extra_label(pd.read_csv(train_csv_path), pd.read_csv(test_csv_path), ['id', 'label'])
    Y = data.pop('attack_cat').values
    Y = preprocessing.LabelEncoder().fit_transform(Y)
    from collections import Counter
    print(Counter(Y))
