from Loader import ObjTrainLoader
from torch.utils.data import DataLoader
from loss import TotalLoss
from models.PhysLSTM import PhysLSTM
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR



train_dir = "path\to\train\folder"
val_dir = "path\to\val\folder"
test_dir = "path\to\test\folder"
batch_size = 2
train_loader_obj = ObjTrainLoader(train_dir)
train_loader = DataLoader(dataset=train_loader_obj, batch_size = batch_size, shuffle=True)

test_loader_obj = ObjTrainLoader(test_dir)
test_loader = DataLoader(dataset=test_loader_obj, batch_size = batch_size, shuffle=True)

val_loader_obj = ObjTrainLoader(val_dir)
val_loader = DataLoader(dataset=val_loader_obj, batch_size = batch_size, shuffle=True)

net = PhysLSTM()
device = "cuda"
net = net.to(device)
epochs = 15
scheduler_step_size = 5

criterion = TotalLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.00001, betas=(0.9, 0.999), eps=1e-8) # , weight_decay=0.00001, betas=(0.9, 0.999), eps=1e-8
scheduler = StepLR(optimizer, step_size=scheduler_step_size, gamma=0.1)

epoch_loss = 1000
iterator = iter(val_loader)
images_val, labels_val = next(iterator)
for epoch in range(epochs):
    net.train()
    train_total_loss = 0
    for i, data in enumerate(train_loader, 0):
        # Get the inputs and labels
        inputs, labels = data
        labels = labels.to(device)
        inputs = inputs.to(device).float()
        inputs = torch.unsqueeze(inputs, 1)

        optimizer.zero_grad()
        outputs = net(inputs)
        outputs = torch.squeeze(outputs, -1)
        outputs_new = outputs[:,40:-20]
        labels_new = labels[:,40:-20]
        loss = criterion(outputs_new, labels_new)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            train_total_loss += loss
            average_loss = train_total_loss/(i+1)
        
        if i % 100 == 0:
            print("loss until sample {} is {}".format(i, average_loss))

    net.eval()
    total_val_loss = 0
    for inputs, labels_val in val_loader:
        inputs = inputs.to(device).float()
        inputs = torch.unsqueeze(inputs, 1)
        #inputs = torch.unsqueeze(inputs, 1)
        with torch.no_grad():
            outputs = net(inputs)
            outputs = torch.squeeze(outputs, -1)
        labels_val = labels_val.to(device)
        outputs_val_new = outputs[:,40:-20]
        labels_val_new = labels_val[:,40:-20]
        l = criterion(outputs_val_new, labels_val_new)
        total_val_loss += l
        
    total_loss = total_val_loss/len(val_loader)
    print("epoch {} is {}".format(epoch+1, total_loss))
    
    if total_loss < epoch_loss:
        weight_name = "Conv_weights.pt"
        torch.save(net.state_dict(), weight_name)
        epoch_loss = total_loss
    scheduler.step()

print('Finished Training')