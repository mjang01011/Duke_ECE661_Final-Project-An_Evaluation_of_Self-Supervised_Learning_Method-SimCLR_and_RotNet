import attacks
import models
from models import process_model_type
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def whitebox_attack(net, whitebox, test_loader, eps, iter, attack_type, device):
    
    whitebox = whitebox.to(device)
    whitebox.eval()
    
    ATK_EPS = eps
    ATK_ITERS = iter
    ATK_ALPHA = 1.85*(ATK_EPS/ATK_ITERS)
    
    whitebox_correct = 0.
    running_total = 0.
    for batch_idx, (data, labels) in enumerate(test_loader):
        data = data.to(device) 
        labels = labels.to(device)
        if attack_type == "random_noise_attack":
            adv_data = attacks.random_noise_attack(model=net, device=device, dat=data, eps=ATK_EPS)
        elif attack_type == "PGD":
            adv_data = attacks.PGD_attack(model=net, device=device, dat=data, lbl=labels, eps=ATK_EPS, alpha=ATK_ALPHA, iters=ATK_ITERS, rand_start=True)
        elif attack_type == "FGM_L2":
            adv_data = attacks.FGM_L2_attack(model=net, device=device, dat=data, lbl=labels, eps=ATK_EPS)
        elif attack_type == "rFGSM":
            adv_data = attacks.rFGSM_attack(model=net, device=device, dat=data, lbl=labels, eps=ATK_EPS)
        else:
            adv_data = attacks.FGSM_attack(model=net, device=device, dat=data, lbl=labels, eps=ATK_EPS)

        assert(torch.max(torch.abs(adv_data-data)) <= (ATK_EPS + 1e-5) )
        assert(adv_data.max() == 1.)
        assert(adv_data.min() == 0.)

        with torch.no_grad():
            whitebox_outputs = whitebox(adv_data)
            _,whitebox_preds = whitebox_outputs.max(1)
            whitebox_correct += whitebox_preds.eq(labels).sum().item()
            running_total += labels.size(0)
        
    whitebox_acc = whitebox_correct/running_total
    
    return whitebox_acc

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, pin_memory=True)

# attack_types = ["random_noise_attack", "PGD", "rFGSM"]
# model_resnet_18 = ["simclr_resnet18_p64_b256","simclr_resnet18_p128_b256","simclr_resnet18_p128_b512"]
# model_resnet_50 = ["simclr_resnet50_p64_b256","simclr_resnet50_p128_b256","simclr_resnet50_p128_b512"]
# projections = [64, 128, 128]
# eps_list = np.linspace(0.0, 0.1, 11)

# for i in range(3):
#     whitebox_accs = []
#     for eps in eps_list:
#         net = process_model_type("simclr_resnet18", load_model="/saved_model/pretrain/" + model_resnet_18[i], simclr_proj_output = projections[i])
#         whitebox = process_model_type("simclr_resnet18", load_model="/saved_model/pretrain/" + model_resnet_18[i], simclr_proj_output = projections[i])
#         whitebox_acc = whitebox_attack(net, whitebox, test_loader, eps, 10, attack_types[0], device)
#         whitebox_acc.append(whitebox_acc)
#     plt.plot(eps_list, whitebox_accs, c='red')
#     plt.title("White Box Attack, Accuracy vs Epsilon" + model_resnet_18[i])
#     plt.xlabel("Epsilon")
#     plt.ylabel("Accuracy")
#     # plt.ylim(0, 1)
#     plt.show()

whitebox_accs = []
eps_list = np.linspace(0.0, 0.1, 11)
for eps in eps_list:
    net = process_model_type("simclr_resnet18", load_model="ECE661/ece661_final_proj/simclr_resnet18_ssl_p128_b256_e1000.pth", simclr_proj_output = 128)
    whitebox = process_model_type("simclr_resnet18", load_model="ECE661/ece661_final_proj/simclr_resnet18_ssl_p128_b256_e1000.pth", simclr_proj_output = 128)
    whitebox_acc = whitebox_attack(net, whitebox, test_loader, eps, 10, attack_types[0], device)
    whitebox_acc.append(whitebox_acc)
plt.plot(eps_list, whitebox_accs, c='red')
plt.title("White Box Attack, Accuracy vs Epsilon" + model_resnet_18[i])
plt.xlabel("Epsilon")
plt.ylabel("Accuracy")
    # plt.ylim(0, 1)
plt.show()
