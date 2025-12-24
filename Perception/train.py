
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random
import numpy as np
import os
import tqdm
import math
import json

from torch.utils.data import DataLoader
from Perception.utils.dataloader import LaneDataset
from Perception.model.unet import UNet
from Perception.utils.loses import dice_loss

device = "cuda"if torch.cuda.is_available() else "cpu"

use_pretrained = None
torch.manual_seed(250)
random.seed(2)


# ----------------- Collect all the file names into two lists ------------------

base_path = "./data/Dataset"
json_name = "/data.json"
if os.path.exists(base_path+json_name):
    f = open(base_path+json_name)
    dataset = json.load(f)
    f.close()
    train_images = dataset["train_images"]
    train_labels = dataset["train_labels"]

    valid_images = dataset["valid_images"]
    valid_labels = dataset["valid_labels"]
else:
    print("No dataset json file first create it !!")
    exit()

 
    
print(len(train_images),len(train_labels))
print(len(valid_images),len(valid_labels))

transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((180,330)),
                                    transforms.ToTensor()])

trainset = LaneDataset(train_images,train_labels,prob=0.15,transforms=transform)
validset = LaneDataset(valid_images,valid_labels,prob=0.6,transforms=transform)


# lower batch size if gpu memory is insufficient
trainloader = DataLoader(trainset,
                        batch_size=32,
                        num_workers=2)

validloader = DataLoader(validset,
                        batch_size=32,
                        num_workers=2)

# ---------------------- Initialize the training loop --------------------------
l_rate = 0.1
momentum = 0.9
num_epochs = 40


save_dict = {
    "epochs" : [],
    "train_loss" : [],
    "train_error" : [],
    "val_loss" : [],
    "val_error" : [],
    "lr_vals" : [],
    "model" : [],
    "min_loss" : np.inf,
    "scheduler" : [],
}

model = UNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(),lr=l_rate,momentum=momentum)
# optimizer = optim.Adam(unet.parameters(),lr=l_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5)


if(use_pretrained != None):
    # save_dict = torch.load("./trained_model_v1.pt")
    save_dict = torch.load(use_pretrained)
    model.load_state_dict(save_dict["model"])
    scheduler.load_state_dict(save_dict["scheduler"])


# save_dict["epochs"].append(save_dict["epochs"][-1] + 1) 


model = model.to(device)

if(save_dict["epochs"] == []):
    start_epoch = 0 
else:  
    save_dict["epochs"].append(save_dict["epochs"][-1] + 1) 
    start_epoch = save_dict["epochs"][-1]

for e in range(start_epoch,num_epochs):
    
    print("Epoch: {}".format(e+1))
    
    total_train_loss = 0
    total_val_loss = 0
    total_train_error = 0
    total_val_error = 0
    num_train_iterations = 0
    num_val_iterations = 0

    model.train()
    with tqdm.tqdm(trainloader, unit="batch") as tepoch:
        for i,data in enumerate(tepoch,0):
            images,labels = data
            
            images = images.to(device)
            labels = labels.to(device)

         
            optimizer.zero_grad()
            out = model(images)
            pred = torch.sigmoid(out)

            # Check if the mask is truly binary
            test_label = labels.detach().cpu().numpy()
            num_not_binary = np.where(((test_label>0)&(test_label<1)|(test_label>1)),1,0).sum()

            # For calculating error
            pred_np = pred.detach()
            labels_np = labels.detach()

            masked_pred = (pred_np>0.5).int()
            correct = torch.sum(torch.bitwise_and(masked_pred,labels_np.type(torch.int32))).item()
            incorrect = torch.sum(torch.bitwise_xor(masked_pred,labels_np.type(torch.int32))).item()

            # Debugging -----------------------------------------------
            # if (labels_np==1).sum().item() == 0:
            #     print((labels_np>0).sum().item())
            #     plt.figure()
            #     plt.subplot(1,2,1)
            #     plt.imshow(images.detach().cpu().numpy().squeeze().transpose(1,2,0))
            #     plt.subplot(1,2,2)
            #     plt.imshow(labels_np.numpy().squeeze())
    #         print(labels)

            error = incorrect/(correct+incorrect)
            loss = criterion(out,labels) + dice_loss(masked_pred,labels)*math.exp(error)
            loss.backward()
            optimizer.step()


            total_train_loss += loss.item()
            total_train_error += error
            num_train_iterations += 1

    save_dict["train_error"].append(total_train_error/num_train_iterations)
    save_dict["train_loss"].append(total_train_loss/num_train_iterations)

    print("Training Error: {} | Training Loss: {} | Number of non-binary: {}".format(total_train_error/num_train_iterations,total_train_loss/num_train_iterations,num_not_binary))
    
    with torch.no_grad():
        model.eval()
        
        with tqdm.tqdm(validloader, unit="batch") as tepoch:
            for i,data in enumerate(tepoch,0):


                images,labels = data
                images = images.to(device)
                labels = labels.to(device)

                out = model(images)
                pred = torch.sigmoid(out)

                # Check if the mask is truly binary
                test_label = labels.detach().cpu().numpy()
                num_not_binary = np.where(((test_label>0)&(test_label<1)|(test_label>1)),1,0).sum()

                # For calculating error
                pred_np = pred.detach()
                labels_np = labels.detach()

                masked_pred = (pred_np>0.5).int()

                correct = torch.sum(torch.bitwise_and(masked_pred,labels_np.type(torch.int32))).item()
                incorrect = torch.sum(torch.bitwise_xor(masked_pred,labels_np.type(torch.int32))).item()
                error = incorrect/(correct+incorrect)

                loss = criterion(out,labels) + dice_loss(masked_pred,labels)*math.exp(error)

                total_val_loss += loss.item()
                total_val_error += error
                num_val_iterations += 1

        
        save_dict["val_error"].append(total_val_error/num_val_iterations)
        save_dict["val_loss"].append(total_val_loss/num_val_iterations)
    
    scheduler.step(total_val_loss/num_val_iterations)
    
    save_dict["lr_vals"].append(optimizer.param_groups[0]['lr'])
    
    print("Validation Error: {} | Validation Loss: {} | Number of non-binary: {}".format(total_val_error/num_val_iterations,total_val_loss/num_val_iterations,num_not_binary))

    save_dict["epochs"].append(e+1)

    if (save_dict["val_loss"][-1] < save_dict["min_loss"]) and (e > 4):
        print("Saved epoch {}".format(e+1))
        save_dict["model"] = model.state_dict()
        save_dict["scheduler"] = scheduler.state_dict()
        save_dict["min_loss"] = save_dict["val_loss"][-1]

        torch.save(save_dict,"epoch_%s.pt"%save_dict["epochs"][-1])
        



# ------------------ Plot the training and validation curves -------------------
fig = plt.figure()
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

ax1.plot(save_dict["epochs"],save_dict["train_error"],label="Training")
ax1.plot(save_dict["epochs"],save_dict["val_error"],label="Validation")
ax1.set_title("Model Error Curves")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Error")
ax1.legend()

ax1.plot(save_dict["epochs"],save_dict["train_loss"],label="Training")
ax1.plot(save_dict["epochs"],save_dict["val_loss"],label="Validation")
ax2.set_title("Model Loss Curves")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()

ax3.plot(save_dict["epochs"],save_dict["lr_vals"],label="Learning Rate")
ax3.set_title("Model LR")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Learning Rate")
ax3.legend()

fig.tight_layout()
plt.show()

save_dict["model"] = model.state_dict()
save_dict["scheduler"] = scheduler.state_dict()
torch.save(save_dict,"a.pt")

# Loading model weights
# unet.load_state_dict(torch.load("Some Path"))