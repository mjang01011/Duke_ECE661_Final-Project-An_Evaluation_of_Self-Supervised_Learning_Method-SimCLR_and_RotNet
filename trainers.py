import tqdm
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
from torch.nn.parallel import DataParallel

class SSLTrainer(): # training encoder for self-supervised without linear evaluation
    def __init__(self, model, optimizer, scheduler, criterion, steps, model_dir, model_title):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        self.model = model.to(self.device)

        self.optim = optimizer 
        self.criterion = criterion 
        self.steps = steps
        self.scheduler = scheduler

        self.model_dir = model_dir
        self.model_title = model_title

    def batch_compute(self, batch, train=True):
        data = batch[0].to(self.device)
        label = batch[1].to(self.device)
        output = self.model(data)
        loss = self.criterion(output)

        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()
        
        return {"loss": loss, "output": output, "label": label}

    def train(self, train_loader, valid_loader, verbose=False, log=False):
        self.model.train()
        step = 0
        while step < self.steps:
            for _, batch in enumerate(tqdm.tqdm(train_loader, desc="Training...")):
                ret = self.batch_compute(batch)
                loss = ret["loss"].item()
                
                if self.model_dir != None and (step + 1) % 500 == 0:
                    state = {'state_dict': self.model.encoder.state_dict(),
                            'steps': step}
                    save_path = f"{self.model_dir}/{self.model_title}_{step+1}.pth"
                    torch.save(state, save_path)
                if log:
                    wandb.log({"step": step, "train_loss": loss })
                
                if verbose:
                    print(f"Step: {step}. Loss: {loss}")
                step += 1
                if step > self.steps:
                    break

    def evaluate(self, args):
        return "No evaluation"

class ClassifierTrainer(): # for training with linear evaluation (semi supervised and supervised)
    def __init__(self, model, optimizer, scheduler, criterion, epochs, model_dir=None, model_title=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        self.model = model.to(self.device)

        self.optim = optimizer 
        self.criterion = criterion 
        self.epochs = epochs
        self.scheduler = scheduler

        self.model_dir = model_dir
        self.model_title = model_title

    def batch_compute(self, batch, train=True):
        data = batch[0].to(self.device)
        label = batch[1].to(self.device)
        output = self.model(data)
        loss = self.criterion(output, label)

        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()
        
        return {"loss": loss, "output": output, "label": label}
    
    def train_epoch(self, dataloader):
        running_loss = 0
        correct = 0
        for _, batch in enumerate(tqdm.tqdm(dataloader, desc="Training...")):
            ret = self.batch_compute(batch)
            running_loss += ret["loss"].item()
            correct += torch.eq(ret['output'].max(1, keepdim=True)[1].flatten(), ret['label']).sum().item()
        loss = running_loss/len(dataloader)
        acc = correct / len(dataloader.dataset)
        return {"loss": loss, "acc": acc}

    def train(self, train_loader, valid_loader, verbose=False, log=False):
        self.model.train()
        best_accuracy = 0
        for epoch in range(self.epochs):
            train_ret = self.train_epoch(train_loader)
            valid_ret = self.evaluate(valid_loader)

            if self.model_dir != None and valid_ret['top1'] > best_accuracy:
                state = {'state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'valid_top1': valid_ret["top1"],
                    'valid_auc': valid_ret["auc"],
                    'valid_loss': valid_ret["loss"]
                    }
                save_path = f"{self.model_dir}/{self.model_title}.pth"
                torch.save(state, save_path)
            if log:
                wandb.log({"epoch": epoch,
                           "train_loss": train_ret["loss"],
                           "train_top1": train_ret["acc"],
                           "valid_loss": valid_ret["loss"],
                           "valid_top1": valid_ret["top1"],
                           "valid_auc": valid_ret["auc"],
                           })
            if verbose:
                if epoch % 1 == 0: # NOTE This should not be 1
                    print(f"Epoch: {epoch}, \nTrain: {train_ret}, \nVal: {valid_ret}")
        return 0

    def evaluate(self, dataloader, verbose=False):
        self.model.eval()
        y_true = []
        y_output = []
        running_loss = 0
        with torch.no_grad():
            for idx, batch in enumerate(tqdm.tqdm(dataloader, desc="Evaluating...")):
                ret = self.batch_compute(batch, train=False)
                running_loss += ret["loss"]
                y_true.extend(ret["label"].cpu())
                y_output.append(F.softmax(ret["output"], dim=1).cpu()) # NOTE Apply softmax so everything sum to 1

        y_true = np.squeeze(np.array(y_true))
        y_output = np.concatenate(y_output, axis=0)
        loss = running_loss/len(dataloader)
        auc = utils.get_auc_score(y_true, y_output)
        top1 = utils.get_topk_accuracy(y_true, y_output, 1)
        top2 = utils.get_topk_accuracy(y_true, y_output, 2)
        top3 = utils.get_topk_accuracy(y_true, y_output, 3)
        return {"loss": loss, "auc": auc, "top1": top1, "top2": top2, "top3": top3}
        




class RotNetTrainer(): # training encoder for self-supervised without linear evaluation
    def __init__(self, model, optimizer, scheduler, criterion, steps, model_dir, model_title):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        self.model = model.to(self.device)

        self.optim = optimizer 
        self.criterion = criterion 
        self.steps = steps
        self.scheduler = scheduler

        self.model_dir = model_dir
        self.model_title = model_title

    def batch_compute(self, batch, train=True):
        data = batch[0].to(self.device)
        label = batch[1].to(self.device)
        output, rot_lab = self.model(data)
        loss = self.criterion(output, rot_lab.to(self.device))

        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()

        return {"loss": loss, "output": output, "label": label, "rotate_label": rot_lab}

    def train(self, train_loader, valid_loader, verbose=False, log=False):
        self.model.train()
        step = 0
        while step < self.steps:
            for _, batch in enumerate(tqdm.tqdm(train_loader, desc="Training...")):
                ret = self.batch_compute(batch)
                loss = ret["loss"].item()

                if self.model_dir != None and (step + 1) % 500 == 0:
                    state = {'state_dict': self.model.encoder.state_dict(),
                            'steps': step}
                    save_path = f"{self.model_dir}/{self.model_title}_s{step+1}.pth"
                    torch.save(state, save_path)
                if log:
                    wandb.log({"steps": step, "train_loss": loss })

                
                step += 1
                if step > self.steps:
                    break
                # if verbose:
                #         print(f"Step: {step}. Loss: {loss}")
            
            # valid_ret = self.evaluate(valid_loader)
            # if verbose:
            #     print(f"Valid: {valid_ret}")

    def evaluate(self, dataloader, verbose=False):
        pass
        # # self.model.eval()
        # # y_true = []
        # # y_output = []
        # # running_loss = 0
        # # with torch.no_grad():
        # #     for idx, batch in enumerate(tqdm.tqdm(dataloader, desc="Evaluating...")):
        # #         ret = self.batch_compute(batch, train=False)
        # #         running_loss += ret["loss"].item()
        # #         y_true.extend(ret["rotate_label"].cpu())
        # #         y_output.append(F.softmax(ret["output"], dim=1).cpu()) # NOTE Apply softmax so everything sum to 1

        # # y_true = np.squeeze(np.array(y_true))
        # # y_output = np.concatenate(y_output, axis=0)
        # # loss = running_loss/len(dataloader)
        # # auc = utils.get_auc_score(y_true, y_output)
        # # top1 = utils.get_topk_accuracy(y_true, y_output, 1)
        # return {"loss": loss, "auc": auc, "top1": top1}
        
