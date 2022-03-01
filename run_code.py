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
    name = "Using_Val_full_splitConv_500"
    wandb.run.name = name
    print(name)

    '''print("making tr_data")
    tr_dataset = BasicDataset(train="Train")
    print("saving tr_data")
    torch.save(tr_dataset, "../data/datasets/tr_unprocessed_144x256.pt")
    print("tr_data saved")

    print("making va_data")
    va_dataset = BasicDataset(train="Valid")
    print("saving va_data")
    torch.save(va_dataset, "../data/datasets/va_unprocessed_bot3_144x256.pt")
    print("va_data saved")'''

    print("loading tr_data")
    tr_dataset = torch.load("../data/datasets/tr_unprocessed_bal_144x256.pt")
    print("tr_data loaded")

    print("loading va_data")
    va_dataset = torch.load("../data/datasets/va_unprocessed_bal_bot3_144x256.pt")
    print("va_data loaded")
   
    batch_size = 128
    epochs = 500
    learningrate = 0.001

    wandb.config = {
        "learning_rate": learningrate,
        "epochs": epochs,
        "batch_size": batch_size
    }

    tr_dataset_size = len(tr_dataset)-2
    va_dataset_size = len(va_dataset)-2
    tr_indices = list(range(tr_dataset_size)) # creates a list that creates a list of indices of the dataset    
    va_indices = list(range(va_dataset_size)) # creates a list that creates a list of indices of the dataset    

    np.random.seed(112) #sets how it will be shuffles
    np.random.shuffle(tr_indices) #splits the dataset and assigns them to training and testing datasets
    np.random.shuffle(va_indices) #splits the dataset and assigns them to training and testing datasets

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(tr_indices)
    valid_sampler = SubsetRandomSampler(va_indices)

    train_loader = DataLoader(tr_dataset, batch_size=batch_size, sampler=train_sampler)
    validation_loader = DataLoader(va_dataset, batch_size=batch_size, sampler=valid_sampler)

    print("Train imgs: ", len(train_loader))
    print("Validation imgs: ", len(validation_loader))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = BasicNetwork().to(device)
    print(model)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)

    tr_key = 0
    val_key = 0
    will_save = False
    try:
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")

            if (epoch % int(epochs/10)) == 0:
                will_save = True
            else:
                will_save = False

            training_dicts, train_error, tr_key = train_loop(train_loader, model, loss_fn, optimizer, device, epoch, batch_size, will_save, tr_key)
            valid_dicts, valid_error, val_key = validation_loop(validation_loader, model, loss_fn, device, epoch, batch_size, will_save, val_key)
            if will_save:
                for dict1, dict2 in zip(training_dicts, valid_dicts):
                    wandb.log(dict1)
                    wandb.log(dict2)
            wandb.log({"Epoch Training Loss":train_error, "Epoch Validation Loss": valid_error, "Epoch epoch":epoch})
    finally:
        save_loc = f"../data/networks/model_{name}.pt"
        print(f"saving Network to {save_loc}")
        torch.save(model, save_loc)
        telegram_send.send(messages=[f"Process Completed ... {name}"])