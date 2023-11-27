# paradata.py
#
# data parallelism implementation

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from model import SimpleCNN

import socket # for hostname recognition

def run_data_parallelism(nodes, gpu_choice):
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

    # Environment variables
    rank = int(os.environ['RANK'])
    worldsize = int(os.environ['WORLD_SIZE'])

    print(f"Initializing process group for rank {rank} out of {worldsize} processes.")

    # Choose the backend based on whether GPUs are being used
    backend = "nccl" if gpu_choice != '0' else "gloo"

    # Initialize the distributed environment
    dist.init_process_group(backend, rank=rank, world_size=worldsize)

    print(f"Process group initialized for rank {rank}.")

    # Check if MNIST dataset is already downloaded
    mnist_data_path = '../data/MNIST'
    if not os.path.exists(mnist_data_path):
        print("Downloading MNIST dataset...")
    else:
        print("MNIST dataset already downloaded.")

    # Dataset and DataLoader setup
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=int(nodes), rank=rank)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=64)

    print(f"Rank {rank}: DataLoader set up complete. Starting training.")

    # Set up the model
    device = torch.device("cuda" if gpu_choice != '0' and torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model = nn.parallel.DistributedDataParallel(model)
    
    print(f"Rank {rank}: Model and optimizer set up complete.")

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # Training loop
    model.train()
    for epoch in range(1, 10):
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if rank == 0 and batch_idx % 10 == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(data_loader.dataset)} ({100. * batch_idx / len(data_loader):.0f}%)]\tLoss: {loss.item():.6f}")

    print(f"Rank {rank}: Training complete.")

    # Cleanup
    dist.destroy_process_group()
    print(f"Rank {rank}: Process group destroyed. Exiting.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--gpu_choice', type=str, default='0', help='GPU choice')
    args = parser.parse_args()
    run_data_parallelism(args.nodes, args.gpu_choice)
