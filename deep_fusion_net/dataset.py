
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def loader(path, batch_size=8, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                # transforms.RandomSizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)

def test_loader(path, batch_size=8, num_workers=4, pin_memory=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return data.DataLoader(
        datasets.ImageFolder(path,
                             transforms.Compose([
                                 # transforms.Resize(256),
                                 # transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 normalize,
                             ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)

if __name__=='__main__':
    train_dir = "/home/lpf/PycharmProjects/DataSet/BreakHis64x64/train/"
    test_dir = "/home/lpf/PycharmProjects/DataSet/BreakHis64x64/test/"
    train_loader = loader(train_dir)
    test_loader = test_loader(test_dir)

    for i, (images, labels) in enumerate(train_loader):
        print(i)
        print(labels)
        print(images.shape)
        images = np.array(images)
        new_image = images[0,:,:,:]
        print(new_image.shape)
        break
        if (i == 0):
            break

