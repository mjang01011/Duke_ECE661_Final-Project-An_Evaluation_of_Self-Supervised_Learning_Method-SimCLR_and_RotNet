import models
import torch
import os

for root, _, files in os.walk("saved_model/linear_eval/"):
        for file in files:
            if "ssl" not in file:
                continue
            
            model_type = "resnet18" if "resnet18" in file else "resnet50"
            model_type = f"simclr_{model_type}"
            print(f"{root}{file}")
            if "_proj" in file:
                p = file.split("_")[3][4:]
            else:
                p = file.split("_")[3][1:] 
            if "_proj" in file:
                b = file.split("_")[4][5:]
            else:
                b = file.split("_")[4][1:] 
            
            epoch = file.split(".")[0].split("_")[-1]
            print(p, b, epoch)

            model = models.process_model_type(model_type=model_type, 
                                  load_model="test_linear_eval", 
                                  simclr_proj_output=p)

            load_file = torch.load(f"{root}{file}")
            model.load_state_dict(load_file['state_dict'])
            state = {'state_dict': model.projector.state_dict(),
                    'epoch': load_file['epoch'],
                    'valid_top1': load_file["valid_top1"],
                    'valid_auc': load_file["valid_auc"],
                    'valid_loss': load_file["valid_loss"]
            }
            torch.save(state, f"saved_model/linear_eval/{model_type}_lineval_p{p}_b{b}_e{epoch}.pth")
