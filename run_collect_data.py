import trainers
import models 
import dataloaders
import utils

import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import argparse
import wandb
import os


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data_dir', default="./data")
parser.add_argument('--model_dir', default="saved_model/linear_eval/")
parser.add_argument('--dataset', default="CIFAR10")
parser.add_argument('--semi_supervised', default=False, type=bool)
parser.add_argument('--semi_supervised_label', default=0.1, type=float)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--lr', '--learning_rate', type=float, default=0.1)
parser.add_argument('--wd', '--weight_decay', type=float, default=0)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--criterion', default="CrossEntropyLoss")
parser.add_argument('--optimizer', default="SGD")
parser.add_argument('--scheduler', default="StepLR")
parser.add_argument('--scheduler_step', default=1000)
parser.add_argument('--scheduler_gamma', type=float, default=1)
parser.add_argument('--scheduler_milestones', default=[250, 375])
parser.add_argument('--model_name', default="")
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--wandb', default=True)

def main():
    args = parser.parse_args()
    args.exp_data_dir = "exp_data/"
    utils.set_all_seeds(args.seed)

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.exp_data_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    transform_train, transform_val_test = utils.process_transformation("simclr_eval")
    
    data = []
    if args.dataset == "STL10":
        _, _, _, test_loader = dataloaders.generate_DataLoader(
            args.dataset, args.data_dir,
            transform_train, transform_val_test,
            args.batch_size, args.batch_size,
            num_workers=args.num_workers)
    else:
        train_loader, valid_loader, test_loader = dataloaders.generate_DataLoader(
            args.dataset, args.data_dir,
            transform_train, transform_val_test,
            args.batch_size, args.batch_size,
            num_workers=args.num_workers)
    
    for root, dir, files in os.walk("saved_model/linear_eval/rotnet/b512"):
        for file in files:
            p = file.split("_")[-3][1:]
            b = file.split("_")[-2][1:]
            s = file.split("_")[-1][1:-4]
            print(file)
            args.network = f"rotnet_resnet50"
            args.load_model = f"{root}/{file}"
    
            args.lin_eval_test = True
            net = models.process_model_type(args.network, load_entire=True, load_model=args.load_model, simclr_proj_output=p)

            criterion = utils.process_criterion_type(args.criterion)
            optimizer = utils.process_optimizer_type(args.optimizer, net.projector, args.lr, args.wd, args.momentum)
            scheduler = utils.process_scheduler_type(args.scheduler, optimizer, args.scheduler_step, args.scheduler_gamma, args.scheduler_milestones)

            net.evaluate = True
            trainer = trainers.ClassifierTrainer(
            model=net,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion, 
            epochs=0)
            trainer.model.load_state_dict(torch.load(args.load_model)['state_dict'])
            ret = trainer.evaluate(test_loader)
            print(ret)
            data.append({
                "batch": b,
                "proj": p,
                "steps": s,
                "model": "resnet50",
                "loss": ret['loss'],
                "top1": ret["top1"],
                "top2": ret["top2"],
                "top3": ret["top3"],
                "auc": ret["auc"],
            })

    torch.save(data, "lineval_rotnet_b512.out")
    return 0


if __name__=="__main__":
    main()
