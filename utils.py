###
# Files for helper function
#    - Metrics Function
#    - Process arguments
#    - Misc Function
###
import numpy as np 
import random 
import torch
import torch.nn as nn
import torch.optim as optim 
from sklearn import metrics
import torchvision.transforms as transforms

from modules.nt_xent import NT_Xent

def set_all_seeds(RANDOM_SEED):
    random.seed(RANDOM_SEED)     # python random generator
    np.random.seed(RANDOM_SEED)  # numpy random generator

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_topk_accuracy(y_true, y_score, k=1):
    return metrics.top_k_accuracy_score(y_true, y_score, k=k)

def get_auc_score(y_true, y_prob):
    return metrics.roc_auc_score(y_true, y_prob, multi_class="ovo")

def process_criterion_type(criterion):
    if criterion == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()
    elif criterion == "nt-xent":
        return NT_Xent(0.5)
    else:
        assert False, "Criterion not implemented"

def process_optimizer_type(optimizer, net, lr, wd, momentum):
    if optimizer == "Adam":
        return optim.Adam(net.parameters(), lr=lr)
    elif optimizer == "SGD":
        return optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum)
    elif optimizer == "NSGD": # SGD with Nesterov
        return optim.SGD(net.parameters(), lr=lr, weight_decay=wd, momentum=momentum, nesterov=True)
    else:
        assert False, "Optimizer not implemented"

def process_scheduler_type(scheduler, optimizer, step, gamma, milestones):
    if scheduler == "StepLR":
        return optim.lr_scheduler.StepLR(optimizer, step, gamma)
    elif scheduler == "MultiStepLR":
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)
    else:
        assert False, "Scheduler not implemented"

def process_transformation(transformation):
    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    if transformation == "default":
        train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=[32,32], padding=(4,2)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    elif transformation == "simclr_default":
        train_transform = transforms.ToTensor()
    elif transformation == "simclr_eval":
        ## TODO Not sure whether the transform should be default or with flip/crop
        train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        assert False

    return train_transform, val_test_transform
