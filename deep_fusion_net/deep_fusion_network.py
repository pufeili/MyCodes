import torch
import torch.nn as nn
from torch.autograd import Variable
from torchsummary import summary
import dataset
import model_generator as mg
import argparse

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")

parser.add_argument("-s", "--sample_num_per_class", type=int, default=1)
parser.add_argument("-b", "--batch_num_per_class", type=int, default=19)
parser.add_argument("-e", "--episode", type=int, default=25)
parser.add_argument("-t", "--test_episode", type=int, default=1000)
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
args = parser.parse_args()

# Hyper Parameters
MAX_EPOCH = args.episode
LEARNING_RATE = args.learning_rate

SAMPLE_NUM_PER_CLASS = 1
BATCH_NUM_PER_CLASS = args.batch_num_per_class

TEST_EPISODE = args.test_episode


def main():

    # step 1: loading data and writing instructions
    print("== 1 == loading data and writing instructions...")
    train_dir = "./datas/BreakHis64x64/train/"
    test_dir = "./datas/BreakHis64x64/test/"
    train_loader = dataset.loader(train_dir)
    test_loader = dataset.test_loader(test_dir)

    filename = './logs/test.txt'
    with open(filename, 'a') as f:
        f.write('\nusing BreakHis64x64 datasets!! \
                 \nbatch_size = 16 \
                 \nnumber of benign equals to malignant!!\n')

    # step 2: init neural networks
    print("== 2 == init neural networks...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    my_model = mg.resnet34(pretrained=False).to(device)
    # summary(my_model,(3,64,64))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_model.parameters(),
                                 lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-08)
    '''
    **********
    need a function to change learning rate!
    need codes for load exiting models!
    **********
    '''
    # step 3 : building graph
    print("== 3 == training...")
    for epoch in range(MAX_EPOCH):
        correct = 0
        total = 0
        loss_mean = 0
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            optimizer.zero_grad()
            outputs = my_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Statistical classification
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().squeeze().sum().numpy()
            loss_mean += loss.item()

            # print result every 500*batch_size
            if (i + 1) % 20 == 0:
                print("Epoch[%d/%d]===Iter[%d/%d]===train_acc:%.2f%%" % ( \
                    epoch + 1, MAX_EPOCH, i + 1, 7500, 100 * correct / total))

        # validation test
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        iter_num = 0
        t_loss_mean = 0
        for iter_num, (t_images, t_labels) in enumerate(test_loader):
            t_images = Variable(t_images.to(device))
            t_labels = Variable(t_labels.to(device))
            t_outputs = my_model(t_images)
            _, predicted = torch.max(t_outputs.data, 1)
            # test_loss = criterion(Outputs, LabelS)
            # t_loss_mean += test_loss.item()

            TP += ((predicted == 0) & (t_labels == 0)).cpu().sum().numpy()
            TN += ((predicted == 1) & (t_labels == 1)).cpu().sum().numpy()
            FN += ((predicted == 1) & (t_labels == 0)).cpu().sum().numpy()
            FP += ((predicted == 0) & (t_labels == 1)).cpu().sum().numpy()
        acc = (TP + TN) / (TP + TN + FP + FN)
        sen = TP / (TP + FN)
        spe = TN / (TN + FP)

        print("Epoch [%d/%d],train_loss %.4f train_acc [%.2f%%]===val_acc[%.2f%%] sen:%.2f%% spe:%.2f%%" \
              % (epoch + 1, MAX_EPOCH, loss_mean / i, 100 * correct / total, 100 * acc, sen * 100, spe * 100))

        filename = './logs/test.txt'
        with open(filename, 'a') as f:
            print("Epoch [%d/%d],train_loss %.4f train_acc [%.2f%%]===val_acc:[%.2f%%] sen:%.2f%% spe:%.2f%%" \
                  % (epoch + 1, MAX_EPOCH, loss_mean / i, 100 * correct / total, 100 * acc, 100 * sen, 100 * spe),
                  file=f)
    '''
    need compare test_acc and last_acc to save model!
    '''
    torch.save(my_model, './models/test.pth')


if __name__ == '__main__':
    main()
