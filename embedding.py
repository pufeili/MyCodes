

import torch
import torch.nn as nn
import numpy as np
import math
import dataset as dset

def EuclideanDistances(A, B):
    BT = B.transpose()# vecProd = A * BT
    vecProd = np.dot(A,BT)
    # print(vecProd)

    SqA =  A**2
    # print(SqA)
    # sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqA = np.sum(SqA, axis=1)
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))

    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0
    ED = np.sqrt(SqED)
    return ED

def Vector_ED(vector_A,vector_B):
    return torch.sqrt(torch.sum(torch.pow(vector_A - vector_B, 2)))
    # results=np.sqrt(np.power(tempA, 2).sum() + np.power(tempB, 2).sum() - 2 * np.dot(tempA, tempB.T)).sum()
    # return results

def Gaussian(vector_A,vector_B,t):
    temp = Vector_ED(vector_A,vector_B)
    result = torch.exp(-temp/t)
    return result

def EmbeddingCost(x_inputs,hidden_output,guassian_t= 1):
    N = x_inputs.size(0)
    inputs = x_inputs.view(N,-1)
    outputs = hidden_output.view(N,-1)
    Jm = 0
    for i in range (N):
        for j in range (N):
            Jm += Gaussian(inputs[i],inputs[j],guassian_t)*Vector_ED(outputs[i],outputs[j])
    Jm = Jm/(N*N)
    return Jm
'''
tensor([[0.5547, 0.6056],
        [0.5667, 0.5390],
        [0.5442, 0.4875],
        [0.5302, 0.4113],
        [0.3417, 0.4174],
        [0.5727, 0.6811],
        [0.4775, 0.6299],
        [0.5656, 0.6574]], grad_fn=<SigmoidBackward>)

'''

if __name__ == '__main__':
    # # train_dir = "./BreakHis64x64/train/"
    # test_dir = "./BreakHis64x64/test/"
    # # train_loader = dset.loader(train_dir)
    # test_loader = dset.test_loader(test_dir)
    #
    # for image,label in  (test_loader):
    #     print(image.shape[0])
    #     break

    # flag = True
    flag = False
    if flag:
        x = torch.tensor([[0.5547, 0.6056],
            [0.5667, 0.5390],
            [0.5442, 0.4875],
            [0.5302, 0.4113],
            [0.3417, 0.4174],
            [0.5727, 0.6811],
            [0.4775, 0.6299],
            [0.5656, 0.6574]],requires_grad=True)
        # b = math.sqrt(math.pow(x[0], 2).sum() + math.pow(x[1], 2).sum() - 2 * np.dot(x[0], x[1].T)).sum()
        # print(torch.pow((x[0] - x[1]),2),x[0] - x[1])
        # b = math.sqrt(math.pow((x[0] - x[1]),2))
        # print(b,b.requires_grad)

        Jm = 0
        gs = 0
        for i in range (8):
            for j in range (8):
                gs += Gaussian(x[i],x[j],0.00001)
                Jm += Vector_ED(x[i],x[j])
        print(Jm,gs)

    # flag = True
    flag = False
    if flag:
        a = torch.tensor([1, 2, 3.], requires_grad=True)
        out = a.sigmoid()
        c = out.data  # 需要走注意的是，通过.data “分离”得到的的变量会和原来的变量共用同样的数据，而且新分离得到的张量是不可求导的，c发生了变化，原来的张量也会发生变化
        # c.zero_()  # 改变c的值，原来的out也会改变
        print(c.requires_grad)
        print(c)
        print(out.requires_grad)
        print(out)
        print("----------------------------------------------")
        out.sum().backward() # 对原来的out求导，
        print(a.grad)  # 不会报错，但是结果却并不正确

    flag = True
    # flag = False
    if flag:
        a = torch.tensor([1, 2, 3.], requires_grad=True)
        out = a.sigmoid()
        c = out.detach()  # 需要走注意的是，通过.detach() “分离”得到的的变量会和原来的变量共用同样的数据，而且新分离得到的张量是不可求导的，c发生了变化，原来的张量也会发生变化
        c.zero_()  # 改变c的值，原来的out也会改变
        print(c.requires_grad)
        print(c)
        print(out.requires_grad)
        print(out)
        print("----------------------------------------------")

        print(out.sum().requires_grad)
        out.sum().backward()  # 对原来的out求导，
        print(a.grad)  # 此时会报错，错误结果参考下面,显示梯度计算所需要的张量已经被“原位操作inplace”所更改了。






