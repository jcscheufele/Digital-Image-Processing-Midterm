from dataset import BasicDataset
from network import BasicNetwork, train_loop, test_loop
#from conv_network import BasicNetwork, train_loop, test_loop
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import torch
import wandb
import telegram_send

if __name__ == "__main__":
    wandb.init(project="CS545_Midterm", entity="jcscheufele")
    name = "Testing Images Small data 2"
    wandb.run.name = name

    '''print("making tr_data")
    tr_dataset = BasicDataset(train=True)
    print("saving tr_data")
    torch.save(tr_dataset, "../data/datasets/tr_unprocessed_240x426.pt")
    print("tr_data saved")

    print("making te_data")
    te_dataset = BasicDataset(train=False)
    print("saving te_data")
    torch.save(te_dataset, "../data/datasets/te_unprocessed_240x426.pt")
    print("te_data saved")'''

    print("loading tr_data")
    tr_dataset = torch.load("../data/datasets/tr_unprocessed_240x426.pt")
    print("tr_data loaded")

    '''print("loading te_data")
    te_dataset = torch.load("../data/datasets/te_unprocessed_240x426.pt")
    print("te_data loaded")'''
    

    #print(len(dataset))
    #print(dataset.X)

    shuffle = True
    batch_size = 128
    epochs = 11
    learningrate = 0.001
    validation_split = 0.3

    wandb.config = {
        "learning_rate": learningrate,
        "epochs": epochs,
        "batch_size": batch_size
    }

    tr_dataset_size = len(tr_dataset)-2
    indices = list(range(tr_dataset_size)) # creates a list that creates a list of indices of the dataset
    split = int(np.floor(validation_split * tr_dataset_size))
    
    if shuffle: #if shuffle is chosen, it will shuffle before it is split
        np.random.seed(112) #sets how it will be shuffles
        np.random.shuffle(indices) #splits the dataset and assigns them to training and testing datasets
    
    train_indices, val_indices = indices[split:], indices[:split]
    '''te_dataset_size = len(te_dataset)-2
    te_indices = list(range(te_dataset_size)) # creates a list that creates a list of indices of the dataset
    
    if shuffle: #if shuffle is chosen, it will shuffle before it is split
        np.random.seed(42) #sets how it will be shuffles
        np.random.shuffle(te_indices) #splits the dataset and assigns them to training and testing datasets
    '''

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(tr_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(tr_dataset, batch_size=batch_size, sampler=valid_sampler)

    print(len(train_loader))
    print(len(validation_loader))

    #in_features = 
    #out_features = 3
    #print(in_features, out_features)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = BasicNetwork().to(device)
    print(model)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)

    tr_key = 0
    te_key = 0
    will_save = False
    try:
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")

            if (epoch % int(epochs/10)) == 0:
                will_save = True
            else:
                will_save = False

            training_dicts, train_error, tr_key = train_loop(train_loader, model, loss_fn, optimizer, device, epoch, batch_size, will_save, tr_key)
            testing_dicts, test_error, te_key = test_loop(validation_loader, model, loss_fn, device, epoch, batch_size, will_save, te_key)
            if will_save:
                for dict1, dict2 in zip(training_dicts, testing_dicts):
                    wandb.log(dict1)
                    wandb.log(dict2)
            wandb.log({"Epoch Training Loss":train_error, "Epoch Testing Loss": test_error, "Epoch epoch":epoch})
    finally:
        save_loc = f"../data/networks/model_{name}.pt"
        print(f"saving Network to {save_loc}")
        torch.save(model, save_loc)
        telegram_send.send(messages=["Process Completed"])