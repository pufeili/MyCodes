import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# loader函数是载入训练集，并对训练集的图片进行正则化
'''
参数：
    path:数据集路径
    batch_size:每次读入的batch大小
    num_workers:工作的进程
    pin_memory:开启时，内存的Tensor转义到GPU的显存就会更快一些
'''


def loader(path, batch_size=8, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 # transforms.Resize([512, 512]),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)


# test_loader函数是载入测试集，并对训练集的图片进行正则化
def test_loader(path, batch_size=8, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 # transforms.Resize([512, 512]),
                                 # transforms.Resize(256),
                                 # transforms.CenterCrop(224),
                                 transforms.RandomResizedCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)


def data_read(target, batch_size=16, num=1):
    if target == "bk":
        train_path = "/home/li/LPF/DataSet/BreakhisDatabase/new_fold1/train/400X/"
        test_path = "/home/li/LPF/DataSet/BreakhisDatabase/new_fold1/test/400X/"
        train_data = loader(train_path, batch_size)
        test_data = test_loader(test_path, batch_size)
        print("----------------------------------------------------------------")
        print("Break Histopathological Images Loaded!")
        return train_data, test_data

    if target == "bc" and num == 1:
        train_path = "/home/li/LPF/DataSet/Hospital/data1_N_L/train/"
        test_path = "/home/li/LPF/DataSet/Hospital/data1_N_L/test/"
        train_data = loader(train_path, batch_size)
        test_data = test_loader(test_path, batch_size)
        print("----------------------------------------------------------------")
        print("Breast Cancer Images N vs L Loaded!")
        return train_data, test_data
    if target == "bc" and num == 2:
        train_path = "/home/li/LPF/DataSet/Hospital/data2_N_S/train/"
        test_path = "/home/li/LPF/DataSet/Hospital/data2_N_S/test/"
        train_data = loader(train_path, batch_size)
        test_data = test_loader(test_path, batch_size)
        print("----------------------------------------------------------------")
        print("Breast Cancer Images N vs S Loaded!")
        return train_data, test_data
    if target == "bc" and num == 3:
        train_path = "/home/li/LPF/DataSet/Hospital/data3_N_L+S/train/"
        test_path = "/home/li/LPF/DataSet/Hospital/data3_N_L+S/test/"
        train_data = loader(train_path, batch_size)
        test_data = test_loader(test_path, batch_size)
        print("----------------------------------------------------------------")
        print("Breast Cancer Images N vs L+S Loaded!")
        return train_data, test_data


if __name__ == '__main__':
    train, test = data_read('bc', num=3)
    for i, (images, labels) in enumerate(train):
        print(i)
        print(labels)
        print(images.shape)
        break
