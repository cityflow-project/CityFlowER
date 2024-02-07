import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import torch
import torch.optim as optim
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, random_split
from tensorboardX import SummaryWriter
import datetime
import os, sys

test_mode = True
test_path = "PATH.pt"
data_set_mode = "create" # "create", "import"

path = "DATA_PATH"
feature_set = []
label_set = []
train_steps = 20

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

if data_set_mode == "create":

    for filename in os.listdir(path):
        sub_item = pd.read_csv(path+filename)
        for index, row in sub_item.iterrows():
            row = row.to_list()
            feature_set.append(row[0:10])
            label_set.append(row[10:11])
    # cut dataset

    dataset = CustomDataset(torch.FloatTensor(feature_set), torch.FloatTensor(label_set))
    dataset_size = len(dataset)
    test_size = int(0.2 * dataset_size)
    train_size = dataset_size - test_size

    # cut into partion
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # create the train and test loader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8)



class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # 10 input features
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)   # Output one value: speed

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  
        return x

model = RegressionModel()
loss_function = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# start training:

if test_mode:
     
    model_load = RegressionModel() 
    model_load.load_state_dict(torch.load(test_path))

    # evaluate model
    model_load.eval()
    with torch.no_grad():
        total_test_loss = 0
        for data, labels in test_loader:
            outputs = model_load(data.float())
            loss = loss_function(outputs, labels.float())
            total_test_loss += loss.item()

        print(f"Test Loss: {total_test_loss / len(test_loader)}")

else:
    epochs = train_steps

    # final version of training cityflow model
    time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    save_path = "PATH"+time_now
    os.mkdir(save_path)
    os.mkdir(save_path+"/save")


    writer = SummaryWriter(save_path+"/"+'writer/'+time_now)


    for data, labels in train_loader:
        print(data.shape)
        print(labels.shape)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        count = 0
        for data, labels in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(data.float())  # Make sure data is float
            loss = loss_function(outputs, labels.float())

            writer.add_scalar('training loss', loss.item(), epoch * len(train_loader) + count)
            count += 1

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")
        if epoch % 50 ==0:
            torch.save(model.state_dict(), save_path+"/save/model_"+str(epoch)+".pth") 

    torch.save(model.state_dict(), save_path+"/save/model_fial_dict.pth")

    # Model class must be defined somewhere
    model_load = RegressionModel()  # Replace with your model class
    model_load.load_state_dict(torch.load(save_path+'/save/model_fial_dict.pth'))

    # evaluate model
    model_load.eval()
    with torch.no_grad():
        total_test_loss = 0
        for data, labels in test_loader:
            outputs = model(data.float())
            loss = loss_function(outputs, labels.float())
            total_test_loss += loss.item()

        print(f"Test Loss: {total_test_loss / len(test_loader)}")


