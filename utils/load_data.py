import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, STL10
from PIL import Image
import random
import torchvision.transforms as transforms
def load_data(data, bs):

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    if data == 'cifar10':
        train_dataset = datasets.CIFAR10('../Dataset/cifar10/', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10('../Dataset/cifar10/', train=False, download=True, transform=transform)

    elif data == 'stl10':
        train_dataset = datasets.STL10('../Dataset/stl10', split="train", download=True, transform=transform)
        test_dataset = datasets.STL10('../Dataset/stl10', split="test", download=True, transform=transform)
    elif data == 'gtsrb':
        train_dataset = datasets.ImageFolder('../Dataset/GTSRB/Train/', transform = transform)
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [30000, len(train_dataset)-30000] )
    elif data == 'animals10':
        full_dataset = datasets.ImageFolder('../Dataset/ANIMALS10/', transform=transform)
        # 按类别收集索引
        class_indices = {i: [] for i in range(len(full_dataset.classes))}
        for idx, (_, label) in enumerate(full_dataset.samples):
            class_indices[label].append(idx)

        train_indices = []
        test_indices = []
        for label, indices in class_indices.items():
            random.shuffle(indices)
            train_indices.extend(indices[:1000])     
            test_indices.extend(indices[1000:1250])  


        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    print('Train dataset: %d, Test dataset: %d'%(len(train_dataset),len(test_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=5, drop_last=False, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=bs, num_workers=5, drop_last=False, shuffle = False)

    return train_loader,  test_loader


def normalzie(args, x):

    if args.dataset == 'cifar10':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalizer = transforms.Normalize(mean=mean, std=std)
        return normalizer(x)

    elif args.dataset == 'stl10' and args.dataset =='gtsrb':
        mean = (0.44087798, 0.42790666, 0.38678814)
        std = (0.25507198, 0.24801506, 0.25641308)
        normalizer = transforms.Normalize(mean=mean, std=std)
        return normalizer(x)

    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        normalizer = transforms.Normalize(mean=mean, std=std)
        return normalizer(x)

