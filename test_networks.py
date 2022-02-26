from dataset import BasicDataset
from network import BasicNetwork, train_loop, test_loop, validation_loop
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
    name = "Using_Val_full_test_1"
    wandb.run.name = name

    print("making te_data")
    te_dataset = BasicDataset(train="Test")
    print("saving te_data")
    torch.save(te_dataset, "../data/datasets/te_unprocessed_single_144x256.pt")
    print("te_data saved")

    '''print("loading te_data")
    te_dataset = torch.load("../data/datasets/te_unprocessed_single_240x426.pt")
    print("te_data loaded")'''

    batch_size = 128
    epochs = 11
    learningrate = 0.001

    wandb.config = {
        "learning_rate": learningrate,
        "epochs": epochs,
        "batch_size": batch_size
    }
    te_dataset_size = len(te_dataset)-2
    te_indices = list(range(te_dataset_size)) # creates a list that creates a list of indices of the dataset    
    np.random.shuffle(te_indices) #splits the dataset and assigns them to training and testing datasets
    test_sampler = SubsetRandomSampler(te_indices)
    test_loader = DataLoader(te_dataset, batch_size=batch_size, sampler=test_sampler)
    print("Test imgs: ", len(test_loader))

    model = torch.load("../data/networks/model_Testing Images Small data 2.pt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    print(model)
    
    te_key = 0
    will_save = True
    loss_fn = nn.BCELoss()
    try:
        testing_dicts, test_error, te_key = test_loop(test_loader, model, loss_fn, device, 0, batch_size, will_save, te_key)
        wandb.log(testing_dicts)
        wandb.log({"Epoch Training Loss":test_error})
    finally:
        telegram_send.send(messages=[f"Testing Completed ... {name}"])