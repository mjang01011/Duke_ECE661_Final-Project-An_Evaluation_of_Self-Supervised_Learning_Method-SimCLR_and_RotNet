
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.utils.data as data
import torch

from torch.utils.data.distributed import DistributedSampler

from sklearn.model_selection import train_test_split
import numpy as np
# DATASETS = [
    # "MNIST",
    # "FMNIST",
    # "CIFAR10",
    # "ImageNet",
# ]


def generate_DataLoader(dataset, root_dir, train_transform, val_test_transform, 
                        train_batch_size, val_test_batch_size, num_workers, val_size=0.1,
                        download=True, shuffle=True, pin_memory=True, swap_train_test=False):
    ## TODO: Need to figure out whether need validation set
    if dataset == "MNIST":
        train_dataset = datasets.MNIST(root=root_dir, train=True, 
                                       download=download, transform=train_transform)
        valid_dataset = datasets.MNIST(root=root_dir, train=True, 
                                       download=download, transform=val_test_transform)
        test_dataset = datasets.MNIST(root=root_dir, train=False,
                                      download=download, transform=val_test_transform)
    elif dataset == "CIFAR10":
        train_dataset = datasets.CIFAR10(root=root_dir, train=True, 
                                       download=download, transform=train_transform)
        valid_dataset = datasets.CIFAR10(root=root_dir, train=True, 
                                       download=download, transform=val_test_transform)
        test_dataset = datasets.CIFAR10(root=root_dir, train=False,
                                      download=download, transform=val_test_transform)
        
        indices = np.arange(len(train_dataset))
        train_idx, valid_idx = train_test_split(indices, test_size=val_size) # shuffle by default true
    elif dataset == "STL10":
        if swap_train_test:
            train_ssl_dataset = datasets.STL10(root=root_dir, split="unlabeled", transform=train_transform, download=download)
            ## TODO: Need to figure out what transformations for which 
            train_classifier_dataset = datasets.STL10(root=root_dir, split="test", transform=train_transform, download=download)
            
            valid_classifier_dataset = datasets.STL10(root=root_dir, split="test", transform=val_test_transform, download=download)

            test_classifier_dataset = datasets.STL10(root=root_dir, split="train", transform=val_test_transform, download=download)
        else:    
            train_ssl_dataset = datasets.STL10(root=root_dir, split="unlabeled", transform=train_transform, download=download)

            ## TODO: Need to figure out what transformations for which 
            train_classifier_dataset = datasets.STL10(root=root_dir, split="train", transform=train_transform, download=download)
            
            valid_classifier_dataset = datasets.STL10(root=root_dir, split="train", transform=val_test_transform, download=download)

            test_classifier_dataset = datasets.STL10(root=root_dir, split="test", transform=val_test_transform, download=download)
        
    else:
        assert False, "dataset not available"

    if dataset == "STL10":
        indices = np.arange(len(train_classifier_dataset))
        train_classifier_idx, valid_classifier_idx = train_test_split(indices, test_size=val_size) # shuffle by default true

        train_classifier_dataset = data.Subset(train_classifier_dataset, train_classifier_idx)
        valid_classifier_dataset = data.Subset(valid_classifier_dataset, valid_classifier_idx)

        train_ssl_loader = DataLoader(train_ssl_dataset, batch_size=train_batch_size, 
                                pin_memory=pin_memory, num_workers=num_workers,
                                shuffle=True, drop_last=True)
        train_classifier_loader = DataLoader(train_classifier_dataset, batch_size=train_batch_size, 
                                pin_memory=pin_memory, num_workers=num_workers,
                                shuffle=True, drop_last=True)
        valid_classifier_loader = DataLoader(valid_classifier_dataset, batch_size=val_test_batch_size,
            shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
        
        test_classifier_loader = DataLoader(test_classifier_dataset, batch_size=val_test_batch_size, 
            shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
        return train_ssl_loader, train_classifier_loader, valid_classifier_loader, test_classifier_loader
    else:
        indices = np.arange(len(train_dataset))
        train_idx, valid_idx = train_test_split(indices, test_size=val_size)

        train_dataset = data.Subset(train_dataset, train_idx)
        valid_dataset = data.Subset(valid_dataset, valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, 
                                pin_memory=pin_memory, num_workers=num_workers,
                                shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=val_test_batch_size,
            shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=val_test_batch_size, 
            shuffle=False, pin_memory=pin_memory, num_workers=num_workers)
        return train_loader, valid_loader, test_loader
