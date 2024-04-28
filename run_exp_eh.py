import trainers
import models 
import dataloaders
import utils

import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

import wandb

import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--data_dir', default="./data")
parser.add_argument('--model_dir', default=None)
parser.add_argument('--dataset', default="CIFAR10")
parser.add_argument('--train_classifier', default=False, type=bool)
parser.add_argument('--network', default="resnet50")
parser.add_argument('--swap_train_test', default=False, type=bool)
parser.add_argument('--train_supervised_linear', default=False, type=bool)
parser.add_argument('--semi_supervised', default=False, type=bool)
parser.add_argument('--eval_semi', default=False, type=bool)
parser.add_argument('--val_size', default=0.1, type=float)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--lr', '--learning_rate', type=float, default=0.01)
parser.add_argument('--wd', '--weight_decay', type=float, default=0)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--criterion', default="CrossEntropyLoss")
parser.add_argument('--optimizer', default="Adam")
parser.add_argument('--scheduler', default="StepLR")
parser.add_argument('--scheduler_step', default=1000, type=int)
parser.add_argument('--scheduler_gamma', type=float, default=0.1)
parser.add_argument('--scheduler_milestones', default=[250, 375])
parser.add_argument('--wandb', default=False)
parser.add_argument('--model_name', default="")
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--train', type=bool, default=False)
parser.add_argument('--simclr_proj_out', type=int, default=64)
parser.add_argument('--verbose', type=bool, default=False)

def main():
    args = parser.parse_args()

    if args.wandb:
        wandb.login()
        wandb.init(
        # set the wandb project where this run will be logged
        project="duke_ece661_sp24_final_proj",

        # track hyperparameters and run metadata
        config=vars(args)
        )

    utils.set_all_seeds(args.seed)

    os.makedirs(args.data_dir, exist_ok=True)

    if args.model_dir != None:
        os.makedirs(args.model_dir, exist_ok=True)

    args.ssl = False
    args.ssl_eval = False
    if args.load_model != None:
        args.ssl_eval = True
    args.transform = "default"
    if args.network.startswith("simclr"):
        args.ssl = True
        args.transform = "simclr_default"
        if args.load_model != None:
            args.ssl_eval = True
            args.transform = "simclr_eval"
        if args.semi_supervised:
            args.ssl_eval = False
            args.transform = "default"
    
    transform_train, transform_val_test = utils.process_transformation(args.transform)

    if args.dataset == "STL10":
        
        train_ssl_loader, train_classifier_loader, valid_classifier_loader, test_classifier_loader = dataloaders.generate_DataLoader(
            args.dataset, args.data_dir,
            transform_train, transform_val_test,
            args.batch_size, args.batch_size,
            num_workers=args.num_workers,
            val_size=args.val_size,
            swap_train_test=args.swap_train_test)
            
    else:
        train_loader, valid_loader, test_loader = dataloaders.generate_DataLoader(
            args.dataset, args.data_dir,
            transform_train, transform_val_test,
            args.batch_size, args.batch_size,
            num_workers=args.num_workers,
            val_size=args.val_size)


    if args.semi_supervised:
        if args.eval_semi:
            net = models.process_model_type(args.network, load_model=args.load_model, fine_tune=True,
                                    simclr_proj_output=args.simclr_proj_out, load_entire=True)    
        else:
            net = models.process_model_type(args.network, load_model=args.load_model, fine_tune=True,
                                    simclr_proj_output=args.simclr_proj_out)
        # net will include both encoder and linear evaluation
    else:
        net = models.process_model_type(args.network, load_model=args.load_model,
                                    simclr_proj_output=args.simclr_proj_out)

    criterion = utils.process_criterion_type(args.criterion)

    if args.ssl_eval:
        # freezing encoder and training linear evaluation
        optimizer = utils.process_optimizer_type(args.optimizer, net.projector, args.lr, args.wd, args.momentum)
    elif args.train_supervised_linear:
        # only train last layer
        # print(net.fc)
        # input()
        optimizer = utils.process_optimizer_type(args.optimizer, net.fc, args.lr, args.wd, args.momentum)
    else:
        if args.semi_supervised: 
            # train both encoder and linear evaluation (net contains encoder and projector)
            optimizer = utils.process_optimizer_type(args.optimizer, net, args.lr, args.wd, args.momentum)
        else:
            # net only has encoder
            optimizer = utils.process_optimizer_type(args.optimizer, net, args.lr, args.wd, args.momentum)

    scheduler = utils.process_scheduler_type(args.scheduler, optimizer, args.scheduler_step, args.scheduler_gamma, args.scheduler_milestones)

    if args.semi_supervised: ## For both RotNet and SimCLR
        net.evaluate = True
        trainer = trainers.ClassifierTrainer(
        model=net,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion, 
        epoch=args.epochs,
        model_dir=args.model_dir,
        model_title=f"{args.network}_{args.model_name}")
    elif args.train_supervised_linear:
        trainer = trainers.ClassifierTrainer(
        model=net,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion, 
        epochs=args.epochs,
        model_dir=args.model_dir,
        model_title=f"{args.network}_{args.model_name}")
    elif args.ssl:
        if args.ssl_eval == False:
            trainer = trainers.SSLTrainer(
            model=net,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion, 
            steps=args.epochs,
            model_dir=args.model_dir,
            model_title=f"{args.network}_{args.model_name}")
        else:
            net.evaluate = True
            trainer = trainers.ClassifierTrainer(
            model=net,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion, 
            epochs=args.epochs,
            model_dir=args.model_dir,
            model_title=f"{args.network}_{args.model_name}")
    elif args.network.startswith("rotnet"):
        if args.load_model != None:
            net.evaluate = True
            trainer = trainers.ClassifierTrainer(
            model=net,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion, 
            epochs=args.epochs,
            model_dir=args.model_dir,
            model_title=f"{args.network}_{args.model_name}")
        else:
            trainer = trainer = trainers.RotNetTrainer(
                model=net,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion, 
                steps=args.epochs,
                model_dir=args.model_dir,
                model_title=f"{args.network}_{args.model_name}")    
    else:
        trainer = trainers.ClassifierTrainer(
            model=net,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion, 
            epochs=args.epochs,
            model_dir=args.model_dir,
            model_title=f"{args.network}_{args.model_name}")
    


    if args.train:
        if args.dataset == "STL10":
            if args.train_classifier:
                ret = trainer.train(train_classifier_loader, valid_classifier_loader, verbose=args.verbose, log=args.wandb)
            else:
                ret = trainer.train(train_ssl_loader, valid_classifier_loader, verbose=args.verbose, log=args.wandb)
                # encoder trainer does not need validation
        else:
            ret = trainer.train(train_loader, valid_loader, verbose=args.verbose, log=args.wandb)

    ret = trainer.evaluate(test_loader)
    print(ret)

    return 0


if __name__=="__main__":
    main()
