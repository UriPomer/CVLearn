# utils/data_loader.py
import torch
from torchvision import datasets, transforms


def get_data_loaders(batch_size=64, num_workers=2):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])

    train_data = datasets.MNIST(root="./data/", transform=transform, train=True, download=True)
    test_data = datasets.MNIST(root="./data/", transform=transform, train=False)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    return train_loader, test_loader
