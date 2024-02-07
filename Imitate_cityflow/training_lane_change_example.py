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
import torch.nn.functional as F

# data list: 
# step, interval, if_changing,  current_time, last_change_time, cooling_time, is_lane, lane_length, 
# distance_on_lane, gap, vehicle_length, vehicle_max_speed, lane_index, lane_size, if_on_last_Road, 
# if_next_drivable_outer, estimate_gap_outer, if_next_drivable_inner, estimate_gap_inner, which_lane


time_now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

save_path = "PATH/"+time_now
os.mkdir(save_path)
os.mkdir(save_path+"/save")


writer = SummaryWriter(save_path+"/"+'writer/'+time_now)

# selected_path
path = "/home/realCityFlow/realCityFlow/Imitate_cityflow/cityflow_lane_change_selected"

data_set_mode = "create" # "create", "import"

feature_set = []
label_set = []

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
            select_data = [row[1]] + row[3:-1]
            feature_set.append(select_data) #change here
            label_set.append(row[-1])
            # print(row)
    
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


for data, labels in train_loader:
    print(data.shape)
    print(labels.shape)

# Define the neural network class
class LaneChangeNetwork(nn.Module):
    def __init__(self):
        super(LaneChangeNetwork, self).__init__()
        # Layers for feature extraction and classification
        self.layers = nn.Sequential(
            nn.Linear(17, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # Output layer for 3 classes
        )

    def forward(self, x):
        # Forward pass through the network
        return self.layers(x)   # Apply softmax to output

model = LaneChangeNetwork()

# Define the loss function and optimizer
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# start training:

epochs = 300

for epoch in range(epochs):
    model.train()
    total_loss = 0
    count = 0
    for data, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(data.float())  # Make sure data is float
        loss = loss_function(outputs, labels.long())

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
model_load = LaneChangeNetwork()  # Replace with your model class
model_load.load_state_dict(torch.load(save_path+'/save/model_fial_dict.pth'))

# evaluate model
model_load.eval()
    
with torch.no_grad():
    total_test_loss = 0
    for data, labels in test_loader:
        outputs = model(data.float())
        loss = loss_function(outputs, labels.float())
        _, predicted_classes = torch.max(outputs, 1)
        # print(predicted_classes)
        total_test_loss += loss.item()

    print(f"Test Loss: {total_test_loss / len(test_loader)}")


