import torch.nn as nn
import torch.nn.functional as F
import torch


class block(nn.Module):
    def __init__(self, in_channels):
        super(block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=(3, 3), padding=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                               kernel_size=(3, 3), padding=1, stride=1)
    def __call__(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv1(x))
        return x+x2


class FootNet(nn.Module):
    def __init__(self, batch_person, person_file_num):
        super(FootNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(7, 7), padding=3, stride=2)
        self.block1 = block(in_channels=64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.block2 = block(in_channels=128)
        self.block3 = block(in_channels=128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1)
        self.block4 = block(in_channels=256)
        self.block5 = block(in_channels=256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1)
        self.block6 = block(in_channels=256)
        self.batch_person = batch_person
        self.person_file_num = person_file_num
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.block1(x)
        x = F.relu(self.conv2(x))  #升核
        x = F.max_pool2d(x, kernel_size=2, stride=2) #降维
        x = self.block2(x)
        x = self.block3(x)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2) #降维
        x = self.block4(x)
        x = self.block5(x)
        x = F.relu(self.conv4(x))  #保持
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 降维
        x = self.block6(x)
        x = F.avg_pool2d(x, kernel_size=(8, 3))
        x = x.view(self.batch_person, self.person_file_num, 256)
        return x









