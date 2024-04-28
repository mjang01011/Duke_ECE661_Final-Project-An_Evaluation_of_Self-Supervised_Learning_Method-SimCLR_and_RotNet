import torch
import torch.nn as nn
import torch.nn.functional as F

def random_noise_attack(model, device, dat, eps):
    x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
    x_adv = torch.clamp(x_adv.clone().detach(), -2.4291, 2.7537)
    return x_adv

def gradient_wrt_data(model,device,data,lbl):
    dat = data.clone().detach()
    dat.requires_grad = True
    out = model(dat)
    # print(out[0])
    # print(lbl[0])
    loss = F.cross_entropy(out,lbl)
    model.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()


def PGD_attack(model, device, dat, lbl, eps, alpha, iters, rand_start):
    x_nat = dat.clone().detach()
    if rand_start:
        x_adv = random_noise_attack(model, device, x_nat, eps)
    else:
        x_adv = x_nat.clone()
    x_adv = torch.clamp(x_adv.clone().detach(), -2.4291, 2.7537)
    # print(x_nat)
    # print(torch.max(torch.abs(x_adv-x_nat)))
    for _ in range(iters):
        data_grad = gradient_wrt_data(model, device, x_adv, lbl)
        x_adv = x_adv + alpha * torch.sign(data_grad)
        x_adv = torch.clamp(x_adv, -eps+x_nat, eps+x_nat)
        # print(torch.max(torch.abs(x_adv-x_nat)))
        x_adv = torch.clamp(x_adv.clone().detach(), -2.4291, 2.7537)
        # print("1clamp",torch.max(torch.abs(x_adv-x_nat)))
    return x_adv


def FGSM_attack(model, device, dat, lbl, eps):
    x_nat = dat.clone().detach()
    x_adv = torch.clamp(x_nat.clone().detach(), -2.4291, 2.7537)
    data_grad = gradient_wrt_data(model, device, x_adv, lbl)
    x_adv = x_adv + eps * torch.sign(data_grad)
    x_adv = torch.clamp(x_adv, -eps+x_nat, eps+x_nat)
    x_adv = torch.clamp(x_adv.clone().detach(), -2.4291, 2.7537)
    return x_adv


def rFGSM_attack(model, device, dat, lbl, eps):
    x_nat = dat.clone().detach()
    x_adv = random_noise_attack(model, device, x_nat, eps)
    x_adv = torch.clamp(x_nat.clone().detach(), -2.4291, 2.7537)
    data_grad = gradient_wrt_data(model, device, x_adv, lbl)
    x_adv = x_adv + eps * torch.sign(data_grad)
    x_adv = torch.clamp(x_adv, -eps+x_nat, eps+x_nat)
    x_adv = torch.clamp(x_adv.clone().detach(), -2.4291, 2.7537)
    return x_adv


def FGM_L2_attack(model, device, dat, lbl, eps):
    x_nat = dat.clone().detach()
    data_grad = gradient_wrt_data(model, device, x_nat, lbl)
    f_grad = data_grad.view(data_grad.shape[0],-1)
    l2_norm_grad = torch.linalg.norm(f_grad, dim=1).view(-1,1,1,1)
    l2_norm_grad = torch.clamp(l2_norm_grad, min=1e-2)
    p = eps*data_grad/l2_norm_grad
    x_adv = x_nat + p
    x_adv = torch.clamp(x_adv, 0., 1.)
    return x_adv
