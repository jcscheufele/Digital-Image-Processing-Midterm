from collections import OrderedDict
from torch import as_tensor, nn, no_grad, cat, flatten, float32, uint8, div, tensor
from torchvision import transforms
import numpy as np
import wandb
from random import sample
import torch

class BasicNetwork(nn.Module):
    def __init__(self, in_features, out_features):
        super(BasicNetwork,self).__init__()
        self.wider_stack = nn.Sequential(OrderedDict([ #Creating an ordered dictionary of the 3 layers we want in our NN
            ('Input', nn.Linear(in_features, 8192)),
            ('Relu 1', nn.ReLU()),
            ('Hidden Linear 1', nn.Linear(8192, 4096)),
            ('Relu 2', nn.ReLU()),
            ('Hidden Linear 2', nn.Linear(4096, 1024)),
            ('Relu 3', nn.ReLU()),
            ('Hidden Linear 3', nn.Linear(1024, 16)),
            ('Relu 4', nn.ReLU()),
            ('Output', nn.Linear(8, out_features))
        ]))
        self.day_convolution = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(814080,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,out_features)
            )
        self.night_convolution = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(814080,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,out_features)
            )
    #Defining how the data flows through the layer, PyTorch will call this we will never have to call it
    def forward(self, x):
        logits1 = self.day_convolution(x[0])
        logits2 = self.night_convolution(x[1])
        logits3 = cat(logits1, logits2)
        logits4 = self.wider_stack(logits3)
        return logits4

def train_loop(dataloader, model, loss_fn, optimizer, device, epoch, bs, will_save, key):
    batches = int(len(dataloader.dataset)*.8/bs)
    cumulative_loss = 0
    ret = []

    for batch, (X, y) in enumerate(dataloader):
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))
        cumulative_loss += loss
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if will_save and (batch % 10 == 0):
            range = sample(list(np.arange(len(X))), min(len(X), 20))
            for idx in range:
                new = torch.reshape(X[idx][:-3], (50,2))
                loc = X[idx][-3:]
                save = {"Train Key": key, "Sample Epoch":epoch,"Sample Training Loss":loss,
                "Diffs": new, 
                "Sample Training Latitude":loc[0], "Sample Training Longitude":loc[1], "Sample Training Altitude":loc[2], 
                "Sample Training Pred Lat": pred[idx][0].item(), "Sample Training Pred Lon": pred[idx][1].item(), "Sample Training Pred Alt": pred[idx][2].item(), 
                "Sample Training Truth Lat": y[idx][0].item(), "Sample Training Truth Lon": y[idx][1].item(), "Sample Training Truth Alt": y[idx][2].item()}
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
def test_loop(dataloader, model, loss_fn, device, epoch, bs, will_save, key):
    batches = int(len(dataloader.dataset)*.2/bs)
    cumulative_loss = 0
    ret = []

    with no_grad():
      for batch, (X, y) in enumerate(dataloader):
        pred = model(X.to(device))
        loss = loss_fn(pred, y.to(device))
        cumulative_loss += loss
        if will_save and (batch % 5 == 0):
            range = sample(list(np.arange(len(X))), min(len(X), 20))
            for idx in range:
                new = torch.reshape(X[idx][:-3], (50,2))
                loc = X[idx][-3:]
                save = {"Test Key": key, "Sample Epoch": epoch, "Sample Testing Loss":loss,
                "Diffs": new, 
                "Sample Testing Latitude":loc[0], "Sample Testing Longitude":loc[1], "Sample Testing Altitude":loc[2], 
                "Sample Testing Pred Lat": pred[idx][0].item(), "Sample Testing Pred Lon": pred[idx][1].item(), "Sample Testing Pred Alt": pred[idx][2].item(), 
                "Sample Testing Truth Lat": y[idx][0].item(),   "Sample Testing Truth Lon": y[idx][1].item(),   "Sample Testing Truth Alt": y[idx][2].item()}
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