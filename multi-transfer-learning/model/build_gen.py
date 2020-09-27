#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
# @FileName  :build_gen.py
# @Time      :2020/9/24 20:33
# @Author    :LPF
"""
import torch
import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'model'))
# print(os.path.join(os.getcwd(), 'model'))
import im2bk
import im2bc
from torchsummary import summary


def generator(source, target, fuse='sum'):
    if source == 'im' and target == 'bk':
        if fuse == 'sum':
            print("---Model from imageNet to BK sum---")
            return im2bk.FuseNet([0, 0, 1, 1])
        else:
            print("---Model from imageNet to BK---")
            return im2bk.TraverseNet([0, 0, 1, 1])

    if source == 'im' and target == 'bc':
        if fuse == 'sum':
            print("---Model from imageNet to BC sum---")
            return im2bc.FuseNet([0, 0, 1, 1])
        else:
            print("---Model from imageNet to BC---")
            return im2bc.TraverseNet([0, 0, 1, 1])

    if source == 'bk' and target == 'bc':
        print("not finish!")
        if fuse == 'sum':
            return im2bk.FuseNet([0, 0, 1, 1])
        else:
            return im2bk.TraverseNet([0, 0, 1, 1])


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    my_model = generator(source='im', target='bc').to(device)
    # my_model = FuseNet([0, 0, 1, 1])
    print(my_model)
    summary(my_model, (3, 512, 512))
    run_code = 0
