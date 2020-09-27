#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
# @FileName  :im2bk.py
# @Time      :2020/9/24 20:38
# @Author    :LPF
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary


class TraverseNet(nn.Sequential):

    def __init__(self, frozen_layers):
        super(TraverseNet, self).__init__()
        self.cnn = models.resnet50(pretrained=True)        
        self.froze_layer = frozen_layers
        self._froze_layers()
        self.cnn.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn.fc = nn.Sequential(nn.Linear(2048, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout())
        self.fc1 = nn.Sequential(nn.Linear(1024, 128),
                                     nn.BatchNorm1d(128),
                                     nn.ReLU(inplace=True),
                                     nn.Dropout(),

                                     nn.Linear(128, 2),
                                     nn.Sigmoid())

    def _froze_layers(self):
        for params in self.cnn.parameters():
            params.requires_grad = False
        if self.froze_layer[0] == 1:
            for params in self.cnn.layer1.parameters():
                params.requires_grad = True
        if self.froze_layer[1] == 1:
            for params in self.cnn.layer2.parameters():
                params.requires_grad = True
        if self.froze_layer[2] == 1:
            for params in self.cnn.layer3.parameters():
                params.requires_grad = True
        if self.froze_layer[3] == 1:
            for params in self.cnn.layer4.parameters():
                params.requires_grad = True

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc1(x)
        return x


class FuseNet(nn.Sequential):

    def __init__(self, frozen_layers):
        super(FuseNet, self).__init__()
        self.cnn = models.resnet50(pretrained=True)
        self.froze_layer = frozen_layers
        self._froze_layers()
        self.cnn.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn.fc = nn.Sequential(nn.Linear(2048, 1024),
                                    nn.BatchNorm1d(1024),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout())
        self.fc1 = nn.Sequential(nn.Linear(1024, 128),
                                 nn.BatchNorm1d(128),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(),

                                 nn.Linear(128, 2),
                                 nn.Sigmoid())
        self.map1 = nn.Conv2d(256, 1024, kernel_size=(1, 1), stride=2)
        self.map2 = nn.Conv2d(512, 1024, (1, 1), stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _froze_layers(self):
        for params in self.cnn.parameters():
            params.requires_grad = False
        if self.froze_layer[0] == 1:
            for params in self.cnn.layer1.parameters():
                params.requires_grad = True
        if self.froze_layer[1] == 1:
            for params in self.cnn.layer2.parameters():
                params.requires_grad = True
        if self.froze_layer[2] == 1:
            for params in self.cnn.layer3.parameters():
                params.requires_grad = True
        if self.froze_layer[3] == 1:
            for params in self.cnn.layer4.parameters():
                params.requires_grad = True

    def forward(self, x):
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x1 = self.cnn.layer1(x)
        x2 = self.cnn.layer2(x1)
        x3 = self.cnn.layer3(x2)

        out = self.maxpool(self.map1(x1)) + self.map2(x2) + x3

        x4 = self.cnn.layer4(out)
        x = self.cnn.avgpool(x4)
        x = torch.flatten(x, 1)
        x = self.cnn.fc(x)
        x = self.fc1(x)

        return x


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # my_model = TraverseNet([0, 0, 1, 1]).to(device)
    my_model = FuseNet([0,0,1,1]).to(device)
    # print(my_model)
    summary(my_model, (3, 512, 512))
    
    x = torch.rand((16, 3, 512, 512)).to(device)
    out = my_model(x)
    print(out.shape)   

    # for pname, p in my_model.named_parameters():
        # # print(pname)
        # if pname == "cnn.layer1.2.conv3.weight":
            # print(p.requires_grad)
        # elif pname == "cnn.layer2.0.conv2.weight":
            # print(p.requires_grad)
        # elif pname == "cnn.layer3.4.conv1.weight":
            # print(p.requires_grad)
        # elif pname == "cnn.layer4.2.conv2.weight":
            # print(p.requires_grad)

    run_code = 0
