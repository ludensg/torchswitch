# paramodel.py
# 
# Contains the implementation of model parallelism using GPipe.

import torch
import torch.nn as nn
import torch.optim as optim
from torchgpipe import GPipe
from model import SimpleCNN
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_gpipe_model(original_model, balance, chunks, devices):
    # Wrap the original model in nn.Sequential
    layers = nn.Sequential(
        original_model.conv1,
        original_model.conv2,
        original_model.fc1,
        original_model.fc2
    )

    # Create the GPipe model
    gpipe_model = GPipe(layers, balance=balance, chunks=chunks, devices=devices)
    return gpipe_model

def train_pipeline(gpipe_model, dataloader, optimizer, criterion, epochs=10):
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()

            # GPipe requires moving data to the appropriate device
            in_device = gpipe_model.devices[0]
            out_device = gpipe_model.devices[-1]
            data, target = data.to(in_device, non_blocking=True), target.to(out_device, non_blocking=True)

            # Forward pass and loss calculation
            outputs = gpipe_model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        # Print statistics
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(dataloader)}, Accuracy: {100 * correct / total}%')


def run_pipeline_parallelism(nodes, gpu_choice):
    # Data loading
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64)

    # Initialize the original model
    original_model = SimpleCNN()

    # Determine chunks based on nodes
    chunks = nodes

    # Device assignment
    if gpu_choice == '1':
        devices = ['cuda:0']
    elif gpu_choice == '2':
        devices = ['cuda:0', 'cuda:1']
    else:
        devices = ['cpu'] * nodes

    # Balance for GPipe
    # Assuming each stage of the model has similar complexity
    balance = [1] * len(devices)

    # Create GPipe model
    gpipe_model = create_gpipe_model(original_model, balance, chunks, devices)

    # Optimizer and criterion
    optimizer = optim.SGD(gpipe_model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    train_pipeline(gpipe_model, dataloader, optimizer, criterion)
