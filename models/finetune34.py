
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary
from torchvision import models
import time
import dataSet

device=torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')

my_model = models.resnet34(pretrained=False)
my_model.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(inplace=True),
    nn.Dropout(),

    nn.Linear(256, 128),
    nn.BatchNorm1d(128),
    nn.ReLU(inplace=True),
    nn.Dropout(),

    nn.Linear(128, 2),
    nn.Sigmoid()
)
my_model = my_model.to(device)
summary(my_model,(3,64,64))


criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer = torch.optim.Adam(my_model.parameters(), lr=lr,betas =(0.9,0.999), eps =1e-08)
Max_epoch = 25

train_dir = "./BreakHis64x64/train/"
test_dir = "./BreakHis64x64/test/"
train_loader = dataSet.loader(train_dir)
test_loader = dataSet.test_loader(test_dir)

filename = './result/trans34exp03.txt'
with open(filename, 'a') as f:
    f.write('\nusing BreakHis64x64 datasets!!\nbatch_size = 32\nnumber of benign equals to malignant!!\n')

for epoch in range(Max_epoch):
    correct = 0
    total = 0
    loss_mean = 0
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())

        optimizer.zero_grad()
        outputs = my_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            lr /= 3
            optimizer = torch.optim.Adam(my_model.parameters(), lr=lr,betas =(0.9,0.999), eps =1e-08)
        # 统计分类情况
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).cpu().squeeze().sum().numpy()
        loss_mean += loss.item()

        if (i+1)%500 == 0:
            print("Epoch[%d/%d]===Iter[%d/%d]===train_acc:%.2f%%"%(epoch+1,Max_epoch,i+1,7500,100*correct/total))
    # print("Epoch [%d/%d] , Train_Loss: %.4f , Train_Acc: %.2f%% " % (epoch + 1,50, loss_mean/i,100*correct/total))

    TP = 0
    TN = 0
    FP = 0
    FN = 0
    iter_num = 0
    t_loss_mean =0
    for iter_num,(ImageS,LabelS) in enumerate (test_loader):
        ImageS = Variable(ImageS.cuda())
        LabelS = Variable(LabelS.cuda())
        Outputs = my_model(ImageS)
        _,predicted = torch.max(Outputs.data,1)
        # test_loss = criterion(Outputs, LabelS)
        # t_loss_mean += test_loss.item()

        TP += ((predicted == 0) & (LabelS == 0)).cpu().sum().numpy()
        TN += ((predicted == 1) & (LabelS == 1)).cpu().sum().numpy()
        FN += ((predicted == 1) & (LabelS == 0)).cpu().sum().numpy()
        FP += ((predicted == 0) & (LabelS == 1)).cpu().sum().numpy()
    acc = (TP + TN) / (TP + TN + FP + FN)
    sen = TP / (TP + FN)
    spe = TN / (TN + FP)
    # print("Test_acc:%.2f%% , Test_loss: %.4f , sen: %.2f , spe: %.2f"%(100*acc,t_loss_mean/iter_num,sen,spe))

    # torch.save(resnet.state_dict(),'resnet.pkl')

    print("Epoch [%d/%d],train_loss %.4f train_acc [%.2f%%]===val_loss %.4f val_acc [%.2f%%]"\
          % (epoch + 1,Max_epoch, loss_mean/i,100*correct/total,t_loss_mean/iter_num,100*acc))

    filename = './result/trans34exp03.txt'
    with open(filename, 'a') as f:
        print("Epoch [%d/%d],train_loss %.4f train_acc [%.2f%%]===val_loss %.4f val_acc [%.2f%%]" \
              % (epoch + 1, Max_epoch, loss_mean / i, 100 * correct / total, t_loss_mean / iter_num, 100 * acc),file=f)

























