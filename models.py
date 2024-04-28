
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np

def process_model_type(model_type, pretrain=False, load_model=None, fine_tune=False, simclr_proj_output = 32, load_entire=False):
    if model_type == "resnet18":
        if pretrain:
            ## Todo, figure out what pretrain weight to use
            net = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.DEFAULT)
        else:
            net = torchvision.models.resnet18(weights=None)

        net.conv1 = nn.Conv2d(3, 64, (3, 3), (1,1))
        net.maxpool = nn.Sequential()
        input_dim = net.fc.in_features
        net.fc = nn.Linear(input_dim, 10)
    elif model_type == "resnet50":
        if pretrain:
            net = torchvision.models.resnet50(
                weights=torchvision.models.ResNet50_Weights.DEFAULT)
        else:
            net = torchvision.models.resnet50(weights=None)
        net.conv1 = nn.Conv2d(3, 64, (3, 3), (1,1))
        net.maxpool = nn.Sequential()
        input_dim = net.fc.in_features
        net.fc = nn.Linear(input_dim, 10)
    elif model_type == "resnet50_lin_transfer":
        # assume pretrained
        encoder = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        input_dim = 2048
        encoder.fc = nn.Linear(input_dim, 10)
        
        # load entire ResNet50
        encoder.load_state_dict(torch.load(load_model)['state_dict'])
        
    elif model_type.startswith("simclr"):
        if model_type == "simclr_resnet50":
            encoder = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            hidden_dim = 2048
        elif model_type == "simclr_resnet18":
            encoder = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
            hidden_dim = 512
        encoder.conv1 = nn.Conv2d(3, 64, (3, 3), (1,1), bias=False)
        nn.init.kaiming_normal_(encoder.conv1.weight, mode="fan_out", nonlinearity="relu")
        encoder.maxpool = nn.Identity()
        encoder.fc =  nn.Identity()
        if load_model == None:
            projector = MLP(hidden_dim, simclr_proj_output)
        else:
            if load_entire is False:
                encoder.load_state_dict(torch.load(load_model)['state_dict'])
            if fine_tune: # want to retrain encoder as well as linear
                projector = Linear(hidden_dim, 10)
            else: # just training linear evaluation
                projector = Linear(hidden_dim, 10)
        net = SimClr(encoder, projector)
        
        if load_entire:
            net.load_state_dict(torch.load(load_model)['state_dict'])
        return net
    elif model_type.startswith("rotnet"):
        if model_type == "rotnet_resnet50":
            encoder = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
            encoder.fc = nn.Identity()
            projector = Linear(2048, 4)
        # self.projector = nn.Linear(2048, 4) 
        net = RotNet(encoder, projector)
        if load_model != None:
            if not load_entire:
                net.encoder.load_state_dict(torch.load(load_model)['state_dict'])
            net.projector = Linear(2048, 10)
        return net
    else:
        assert False, "model not defined"
    
    if load_model != None:
        net.load_state_dict(torch.load(load_model)['state_dict'])
    return net

# TODO: Implement SIMCLR
class SimClr(nn.Module):
    def __init__(self, encoder, projector):
        super().__init__()

        self.encoder = encoder
        self.projector = projector
        self.train_transform = self.get_train_transform()
        self.evaluate = False


    def forward(self, x):
        if self.evaluate:
            return self.projector(self.encoder(x))

        x_1 = self.train_transform(x)
        x_2 = self.train_transform(x)

        h_1 = self.encoder(x_1)
        h_2 = self.encoder(x_2)

        z_1 = self.projector(h_1)
        z_2 = self.projector(h_2)

        z = torch.zeros((z_1.shape[0]*2, z_1.shape[1]), dtype=torch.float)
        idx = 0
        for a,b in zip(z_1, z_2):
            z[idx] = a
            idx += 1 
            z[idx] = b 
            idx += 1
        return z

    def get_train_transform(self):
        s = 1
        random_resize_crop_flip = transforms.Compose([
                    transforms.RandomResizedCrop(size=(32,32), antialias=True),
                    transforms.RandomHorizontalFlip(p=0.5),
                ])
        
        '''From SimCLR Appendix A'''
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distortion = transforms.Compose([
                    rnd_color_jitter,
                    rnd_gray
                ])
        
        # gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        # random_gaussian_blur = transforms.Compose([
        #     transforms.RandomApply([gaussian_blur], p=0.5)        
        # ])

        return transforms.Compose(
            [random_resize_crop_flip, color_distortion])

class RotNet(nn.Module):
    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder 
        self.projector = projector
        # self.encoder = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        # self.encoder.conv1 = nn.Conv2d(3, 64, (3, 3), (1,1), bias=False)
        # nn.init.kaiming_normal_(self.encoder.conv1.weight, mode="fan_out", nonlinearity="relu")
        # self.encoder.maxpool = nn.Identity()
        # self.encoder.fc = nn.Identity()
        # self.projector = nn.Linear(2048, 4) 
        self.evaluate = False

    def forward(self, x):
        if self.evaluate:
            return self.projector(self.encoder(x))
        x_0, x_90, x_180, x_270  = self.get_transform(x)
        batch_x = torch.cat([x_0, x_90, x_180, x_270], 0).to("cuda")
        batch_rot_y = torch.cat((                                                   # batch_rot_y: [bs*4]
            torch.zeros(x.shape[0]),
            torch.ones(x.shape[0]),
            2 * torch.ones(x.shape[0]),
            3 * torch.ones(x.shape[0])
        ), 0).long().to("cuda")
        hidden = self.encoder(batch_x)
        out = self.projector(hidden)
        return out, batch_rot_y

    def get_transform(self, x):
        return TF.rotate(x, 0), TF.rotate(x, 90), TF.rotate(x, 180), TF.rotate(x, 270), 

class MLP(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()

        self.w1 = nn.Linear(hidden_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU(True)
        return
    
    def forward(self, x):
        return self.w2(self.relu(self.w1(x)))

class Linear(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()

        self.fc = nn.Linear(hidden_dim, output_dim)
        return

    def forward(self, x):
        return self.fc(x)
