import torch
import torchvision
import pandas as pd
from torchvision import transforms


def get_celebA_dataset(batch_size, img_size, mini=False):
    image_path = "data/"
    '''
    transformation = transforms.Compose([transforms.Resize(image_size),
                                            transforms.CenterCrop(image_size), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5)), ])

    train_dataset = torchvision.datasets.ImageFolder(image_path, transformation)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    train_indices, test_indices = indices[:10000], indices[200000:]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler)
    '''
    dataset = torchvision.datasets.CelebA(root=image_path, download=True,
                               transform=transforms.Compose([
                                   transforms.Resize((img_size,img_size)),
                                   transforms.CenterCrop(img_size), 
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def get_ffhq_thumbnails(batch_size, image_size):
    image_path = "data/"
    transformation = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.ImageFolder(image_path + 'FFHQ/', transformation)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    train_indices, test_indices = indices[:60000], indices[60000:]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler)
    return train_loader, test_loader

'''
def get_celebA_dataset(batch_size, img_size):
    image_path = "./"
    transformation = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.ImageFolder(image_path + 'celebA', transformation)
    
    indices = pd.read_csv('celebA/list_eval_partition.csv')
    train_indices = list(indices[indices.partition == 0].index)
    validation_indices = list(indices[indices.partition == 1].index)[:10]
    test_indices = list(indices[indices.partition == 2].index)
    
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    validation_sampler = torch.utils.data.SubsetRandomSampler(validation_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                    sampler=validation_sampler)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                              sampler=test_sampler)
    
    return train_loader, validation_loader, test_loader
'''

def get_lsun_dataset(batch_size, image_size, classes) : 
    # 공유 폴더 lsun안의 폴더째로 ../lsun/dataset에 저장
    image_path = "../lsun/dataset"
    dataset = torchvision.datasets.LSUN(root=image_path, classes=[classes], transform=transforms.Compose([
                                            transforms.Resize(image_size),
                                            transforms.CenterCrop(image_size), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5)),  ]))
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def get_mnist_dataset(batch_size, img_size) : 
    image_path = "../dataset"
    dataset = torchvision.datasets.MNIST(root=image_path, download=True,
                               transform=transforms.Compose([
                                   transforms.Grayscale(3),
                                   transforms.Resize((img_size,img_size)),
                                   transforms.CenterCrop(img_size), 
                                   transforms.ToTensor(),
                               ]))
    data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data

def get_mnist_fashion_dataset(batch_size, img_size) : 
    image_path = "../dataset"
    dataset = torchvision.datasets.FashionMNIST(root=image_path, download=True,
                               transform=transforms.Compose([
                                   transforms.Grayscale(3),
                                   transforms.Resize((img_size,img_size)),
                                   transforms.CenterCrop(img_size), 
                                   transforms.ToTensor(),
                               ]))
    data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data

def get_emnist_dataset(batch_size, img_size) : 
    image_path = "../dataset"
    dataset = torchvision.datasets.EMNIST(root=image_path, download=True, split='balanced',
                               transform=transforms.Compose([
                                   transforms.Grayscale(3),
                                   transforms.Resize((img_size,img_size)),
                                   transforms.CenterCrop(img_size), 
                                   transforms.ToTensor(),
                               ]))
    data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data

def get_cifar1_dataset(batch_size, img_size, shuffle=True) : 
    image_path = "../dataset"
    dataset = torchvision.datasets.CIFAR10(root=image_path,  download=True,
                               transform=transforms.Compose([
                                   transforms.Resize((img_size,img_size)),
                                   transforms.CenterCrop(img_size), 
                                   transforms.ToTensor(),
                               ]))
    data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data

if __name__ == '__main__':
    batch_size = 64
    img_size = 32
    train_loader, validation_loader, test_loader = get_celebA_dataset(batch_size, img_size)
    train_loader = get_cifar1_dataset(batch_size, img_size)
