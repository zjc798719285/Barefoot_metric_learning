import torch.nn as nn
import torch.nn.functional as F
import torch

class FootNet(nn.Module):
    def __init__(self, batch_person, person_file_num):
        super(FootNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(7, 7), padding=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)
        self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1)
        self.batch_person = batch_person
        self.person_file_num = person_file_num
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))  #升核
        x = F.max_pool2d(x, kernel_size=2, stride=2) #降维
        x = F.relu(self.conv3(x))  #保持
        x = F.relu(self.conv4(x))  #升核
        x = F.max_pool2d(x, kernel_size=2, stride=2) #降维
        x = F.relu(self.conv5(x))  #保持
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # 降维
        x = F.relu(self.conv6(x))  # 保持
        x = F.avg_pool2d(x, kernel_size=(8, 3))
        x = x.view(self.batch_person, self.person_file_num, 128)
        return x









