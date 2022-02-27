from collections import OrderedDict
from torch import as_tensor, nn, no_grad, cat, flatten, float32, uint8, div, tensor
from torchvision import transforms
import numpy as np
import wandb
from random import sample
import torch

class BasicNetwork(nn.Module):
    def __init__(self):
        super(BasicNetwork,self).__init__()
        self.day_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(36864,1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.night_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(36864,1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU()
        )
        self.wider_stack = nn.Sequential(OrderedDict([ #Creating an ordered dictionary of the 3 layers we want in our NN
            ('Input', nn.Linear(128, 64)),
            ('Relu 1', nn.ReLU()),
            ('Hidden Linear 1', nn.Linear(64, 16)),
            ('Relu 2', nn.ReLU()),
            ('Raw Output', nn.Linear(16, 1)),
            ("Sigmoid", nn.Sigmoid())
        ]))
    #Defining how the data flows through the layer, PyTorch will call this we will never have to call it
    def forward(self, x):
        logits1 = self.day_linear(x[0])
        logits2 = self.night_linear(x[1])
        #print(logits1.shape, logits2.shape)
        logits3 = cat((logits1, logits2), 1)
        #print(logits3.shape)
        logits4 = self.wider_stack(logits3)
        return logits4
        '''logits = self.single_convolution(x)
        logits = self.wider_stack(logits)
        return logits'''

def train_loop(dataloader, model, loss_fn, optimizer, device, epoch, size, will_save, key):
    batches = int(np.floor(size))
    cumulative_loss = 0
    ret = []

    for batch, (X_day, X_night, y) in enumerate(dataloader):
        pred = model((X_day.to(device), X_night.to(device)))
        loss = loss_fn(pred, torch.reshape(y,(y.shape[0],1)).to(device))
        cumulative_loss += loss
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if will_save and (batch == batches):
            range = sample(list(np.arange(len(X_day))), min(len(X_day), 100))
            for idx in range:
                save = {"Train Key": key, "Sample Epoch":epoch,"Sample Training Loss":loss,
                "Sample Training Image Day": wandb.Image(X_day[idx]), "Sample Training Image Night": wandb.Image(X_night[idx]),
                "Sample Training Pred": pred[idx].item(), "Sample Training Truth": y[idx].item()}
                ret.append(save)
                key+=1
            row = f"{epoch} [ {batch}/{batches} ] Loss: {loss} SAVED\n"
            print(row)
        else:
            row = f"{epoch} [ {batch}/{batches} ] Loss: {loss}\n"
            print(row)

    averages_1 = f"End of Testing \n Test Error: \n Testing Avg loss: {(cumulative_loss/batches):>8f}\n"
    print(averages_1)
    return ret, cumulative_loss/batches, key

#Runs through the whole dataset and gives final performace metrics
def test_loop(dataloader, model, loss_fn, device, epoch, size, will_save, key):
    batches = int(np.floor(size))
    cumulative_loss = 0
    ret = []

    with no_grad():
      for batch, (X_day, X_night, y) in enumerate(dataloader):
        pred = model((X_day.to(device), X_night.to(device)))
        loss = loss_fn(pred, torch.reshape(y,(y.shape[0],1)).to(device))
        cumulative_loss += loss
        if will_save and (batch == batches):
            range = sample(list(np.arange(len(X_day))), min(len(X_day), 100))
            for idx in range:
                save = {"Test Key": key, "Sample Epoch":epoch,"Sample Testing Loss":loss,
                "Sample Testing Image Day": wandb.Image(X_day[idx]), "Sample Testing Image Night": wandb.Image(X_night[idx]),
                "Sample Testing Pred": pred[idx].item(), "Sample Testing Truth": y[idx].item()}
                ret.append(save)
                key+=1
            row = f"{epoch} [ {batch}/{batches} ] Loss: {loss} SAVED\n"
            print(row) 
        else:
            row = f"{epoch} [ {batch}/{batches} ] Loss: {loss}\n"
            print(row)

    averages_1 = f"End of Testing \n Test Error: \n Testing Avg loss: {(cumulative_loss/batches):>8f}\n"
    print(averages_1)
    return ret, cumulative_loss/batches, key

def validation_loop(dataloader, model, loss_fn, device, epoch, size, will_save, key):
    batches = int(np.floor(size))
    cumulative_loss = 0
    ret = []

    with no_grad():
      for batch, (X_day, X_night, y) in enumerate(dataloader):
        pred = model((X_day.to(device), X_night.to(device)))
        loss = loss_fn(pred, torch.reshape(y,(y.shape[0],1)).to(device))
        cumulative_loss += loss
        if will_save and (batch == batches):
            range = sample(list(np.arange(len(X_day))), min(len(X_day), 100))
            for idx in range:
                save = {"Valid Key": key, "Sample Epoch":epoch,"Sample Valid Loss":loss,
                "Sample Valid Image Day": wandb.Image(X_day[idx]), "Sample Valid Image Night": wandb.Image(X_night[idx]),
                "Sample Valid Pred": pred[idx].item(), "Sample Valid Truth": y[idx].item()}
                ret.append(save)
                key+=1
            row = f"{epoch} [ {batch}/{batches} ] Loss: {loss} SAVED\n"
            print(row) 
        else:
            row = f"{epoch} [ {batch}/{batches} ] Loss: {loss}\n"
            print(row)

    averages_1 = f"End of Validation \n Validation Error: \n Validation Avg loss: {(cumulative_loss/batches):>8f}\n"
    print(averages_1)
    return ret, cumulative_loss/batches, key