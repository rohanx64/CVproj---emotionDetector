import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super().__init__()
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2,2)
        self.drop = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        s = self.shortcut(x)
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        out = F.relu(s + y)
        return self.drop(self.pool(out))

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.rb1 = ResidualBlock(1, 32)
        self.rb2 = ResidualBlock(32, 64)
        self.rb3 = ResidualBlock(64, 128)
        self.rb4 = ResidualBlock(128, 256)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.drop_fc = nn.Dropout(0.5)
        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.rb1(x)
        x = self.rb2(x)
        x = self.rb3(x)
        x = self.rb4(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop_fc(x)
        return self.out(x)
