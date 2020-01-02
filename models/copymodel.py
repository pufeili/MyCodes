import torch
import torch.nn as nn
import dataSet
from torch.autograd import Variable
import dataload
from torchsummary import summary


class ResBlock(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1,
                               stride=stride,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               padding=1, bias=False)    #padding=same:when s=1,p=(f-1)/2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class MyNet(nn.Module):
    def __init__(self, block, layers, num_classes=2):
        self.inplanes = 64
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2,padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2 )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.avgpool1 = nn.AvgPool2d(7, stride=7)  #输入不同，更改avgpool即可
        self.avgpool2 = nn.AvgPool2d(4)

        self.fc = nn.Linear(256 * block.expansion, num_classes)
        # self.fc1 = nn.Sigmoid(512 * block.expansion, num_classes)
        self.classifier = nn.Sequential(

            nn.Linear(1024,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128,2),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)   #channel 3 -> 64
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool1(x)
        x = self.avgpool2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    # def fit_keras(self,train_x=None,train_y=None,batch_size=None,amodel,
    #         Max_epoch=1,shuffle=True,criterion=None,optimizer=None):
    #     if criterion == None:
    #         criterion = nn.CrossEntropyLoss()
    #     if optimizer == None:
    #         optimizer = torch.optim.Adam(amodel.parameters(), lr=0.001,betas =(0.9,0.999), eps =1e-08)

    def fit_pytorch(self,train_dir,my_model,save_model=True,
                    optimizer=None,criterion=None,Max_epoch=20,lr= 0.001):
        train_loader = dataSet.loader(train_dir)
        if criterion == None:
            criterion = nn.CrossEntropyLoss()
        if optimizer == None:
            optimizer = torch.optim.Adam(my_model.parameters(), lr=0.001,betas =(0.9,0.999), eps =1e-08)
        for epoch in range(Max_epoch):
            total = 0
            correct = 0
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
                    optimizer = torch.optim.Adam(my_model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
                # 统计分类情况
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).cpu().squeeze().sum().numpy()
                loss_mean += loss.item()
        print("Epoch [%d/%d] , Train_Loss: %.4f , Train_Acc: %.2f%% " %(epoch + 1, 50, loss_mean / i, 100 * correct / total))
        if save_model:
            torch.save(my_model,'./my_model.pkl')

if __name__ == '__main__':
    device=torch.device('cuda:0'if torch.cuda.is_available() else 'cpu')
    model30 = MyNet(ResBlock,[3,3,3]).to(device)
    # summary(model30,(3,512,512))
    # newNet = torch.load('acc80.pkl')

    criterion = nn.CrossEntropyLoss()
    lr = 0.001
    optimizer = torch.optim.Adam(model30.parameters(), lr=lr,betas =(0.9,0.999), eps =1e-08)
    Max_epoch = 50

    train_dir = "./BreakHis_orig/train/"
    test_dir = "./BreakHis_orig/test/"

    # train_dir = "./BreakH/train/"
    # test_dir = "./BreakH/test/"

    train_loader = dataSet.loader(train_dir)
    test_loader = dataSet.test_loader(test_dir)

    for epoch in range(Max_epoch):
        correct = 0
        total = 0
        loss_mean = 0
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model30(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # if (epoch + 1) % 20 == 0:
            #     lr /= 3
            #     optimizer = torch.optim.Adam(model30.parameters(), lr=lr,betas=(0.9, 0.999), eps =1e-08)
            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).cpu().squeeze().sum().numpy()
            loss_mean += loss.item()
        # print("Epoch [%d/%d] , Train_Loss: %.4f , Train_Acc: %.2f%% " % (epoch + 1,Max_epoch, loss_mean/(i+1),100*correct/total))

        TP = 0
        TN = 0
        FP = 0
        FN = 0
        t_loss_mean = 0
        iter_num = 0
        for iter_num,(ImageS,LabelS) in enumerate (test_loader):
            ImageS = Variable(ImageS.cuda())
            LabelS = Variable(LabelS.cuda())
            Outputs = model30(ImageS)
            _,predicted = torch.max(Outputs.data,1)

            TP += ((predicted == 0) & (LabelS == 0)).cpu().sum().numpy()
            TN += ((predicted == 1) & (LabelS == 1)).cpu().sum().numpy()
            FN += ((predicted == 1) & (LabelS == 0)).cpu().sum().numpy()
            FP += ((predicted == 0) & (LabelS == 1)).cpu().sum().numpy()
        acc = (TP + TN) / (TP + TN + FP + FN)
        sen = TP / (TP + FN)
        spe = TN / (TN + FP)
        # print("Test_acc:%.2f%% , sen: %.2f , spe: %.2f"%(100*acc,sen,spe))
        # torch.save(resnet.state_dict(),'resnet.pkl')

        print("Epoch [%d/%d],train_loss %.4f train_acc [%.2f%%]===val_loss %.4f val_acc [%.2f%%]"\
              % (epoch + 1,Max_epoch, loss_mean/(i+1),100*correct/total,t_loss_mean/(iter_num+1),100*acc))

        filename = './result/copymodel_new01.txt'
        with open(filename, 'a') as f:
            print("Epoch [%d/%d],train_loss %.4f train_acc [%.2f%%]===val_loss %.4f val_acc [%.2f%%]" \
                  % (epoch+1,Max_epoch, loss_mean/(i+1), 100*correct/total,t_loss_mean/(iter_num+1),100*acc),file=f)

    filename = './result/copymodel_new01.txt'
    with open(filename, 'a') as f:
        f.write('train batch_size==8==test\nHK_origin')








