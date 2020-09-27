#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
 @FileName  :bk2bc.py
 @Time      :2020/9/25 12:50
 @Author    :LPF
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchsummary import summary





if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load('../checkpoints/im2bc_sum_num_data_2_record_num_6.pth').to(device)
    # print(model)
    for pname, p in model.named_parameters():
        print(pname)
        # if pname == "cnn.layer1.2.conv3.weight":
            # print(p.requires_grad)
        # elif pname == "cnn.layer2.0.conv2.weight":
            # print(p.requires_grad)
        # elif pname == "cnn.layer3.4.conv1.weight":
            # print(p.requires_grad)
        # elif pname == "cnn.layer4.2.conv2.weight":
            # print(p.requires_grad)
    run_code = 0
