import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):

        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=6)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class LSTM(nn.Module):
    def __init__(self):

        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=196, hidden_size=70, num_layers=1, batch_first=True)
        self.fc = nn.Linear(70, 10)

    def forward(self, x):
        out, (_, _) = self.lstm(x)
        out = self.fc(out[:, -1, :])

        return out


if __name__ == "__main__":
    import torch
    x = torch.rand(size=(64, 1, 196))
    lstm = LSTM()
    out = lstm(x)
    print(out.shape)
