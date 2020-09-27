#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
 @FileName  :solver.py
 @Time      :2020/9/25 14:20
 @Author    :LPF
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
from dataset import data_read
from model.build_gen import generator


class Solver(object):
    def __init__(self, args, batch_size=16, source='im', target='bc', num_bc=2,
                 learning_rate=0.0002, fuse_mode=None, optimizer='adam'):
        self.max_epoch = args.max_epoch
        self.save_epoch = args.save_epoch
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.source = source
        self.target = target
        self.num_bc = num_bc        
        self.datas, self.datas_test = data_read(target, num=self.num_bc, batch_size=self.batch_size)

        self.Net = generator(source, target, fuse_mode).to(self.device)

        self.optimizer, self.scheduler = self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = learning_rate

    def set_optimizer(self,  which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            optimizer = torch.optim.SGD(self.Net.parameters(),
                                        lr=lr, weight_decay=0.0005,
                                        momentum=momentum)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
            return optimizer, scheduler

        if which_opt == 'adam':            
            optimizer = torch.optim.Adam(self.Net.parameters(),
                                         lr=lr, weight_decay=0.0005, betas=(0.9, 0.999), eps=1e-08)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
            return optimizer, scheduler

    def reset_grad(self):
        self.optimizer.zero_grad()

    def train(self, epoch, record_file=None):
        print("----------------------------------------------------------------")
        criterion = nn.CrossEntropyLoss().to(self.device)
        self.Net.train()

        correct = 0
        total = 0
        loss_mean = 0
        for i, (images, labels) in enumerate(self.datas):
            images = Variable(images.to(self.device))
            labels = Variable(labels.to(self.device))

            self.reset_grad()
            outputs = self.Net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()            

            # Statistical classification
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().squeeze().sum().numpy()
            loss_mean += loss.item()

            # print result every 500*batch_size
            if (i + 1) % 500 == 0:
                print("  Iter[%d------>%d] train_acc:%.2f%%" % (i + 1, 3500, 100 * correct / total))
        # print('lr: ', self.scheduler.get_lr())
        self.scheduler.step()        
        print("Epoch [%2d/%d],Train set: loss[%.4f], acc[%.2f%%]"%(epoch+1, self.max_epoch, loss_mean/i, 100*correct/total))
        if record_file:
            with open(record_file, 'a') as f:
                print("Epoch [%2d/%d],Train set: loss[%.4f], acc[%.2f%%]" % (
                epoch+1, self.max_epoch, loss_mean/i, 100*correct/total), file=f)

    def train_onestep(self):
        pass

    def test(self, epoch, record_file=None, save_model=False):
        criterion = nn.CrossEntropyLoss().to(self.device)
        self.Net.eval()
        # validation test
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        test_loss = 0
        # 对测试集进行测试，并统计结果
        for iter_num, (images, labels) in enumerate(self.datas_test):
            with torch.no_grad():
                images = Variable(images.to(self.device))
                labels = Variable(labels.to(self.device))
            
            outputs = self.Net(images)
            test_loss += criterion(outputs, labels).item()
            
            _, predicted = torch.max(outputs.data, 1)          
            
            TP += ((predicted == 0) & (labels == 0)).cpu().sum().numpy()
            TN += ((predicted == 1) & (labels == 1)).cpu().sum().numpy()
            FN += ((predicted == 1) & (labels == 0)).cpu().sum().numpy()
            FP += ((predicted == 0) & (labels == 1)).cpu().sum().numpy()
        acc = (TP + TN) / (TP + TN + FP + FN)
        sen = TP / (TP + FN)
        spe = TN / (TN + FP)
        loss = test_loss / iter_num
        
        print("              Test  set: loss[{:.4f}], acc[{:.2f}%], sen[{:.2f}%], spe[{:.2f}%]".format(
                loss, acc*100, sen*100, spe*100))

        if record_file:
            with open(record_file, 'a') as f:
                print("              Test  set: loss[{:.4f}], acc[{:.2f}%], sen[{:.2f}%], spe[{:.2f}%]".format(
                        loss, acc*100, sen*100, spe*100), file=f)
        if save_model and epoch % self.save_epoch == 0:
            model_name = record_file.split('/')[2].split('.')[0] + '.pth'
            torch.save(self.Net, './checkpoints/' + model_name)  
            # torch.save(model.state_dict(), checkpoint_dir + model_name +"_model_state_dict.pth")


if __name__ == "__main__":

    run_code = 0
