# model.py

import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer taking a single-channel (grayscale) input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)  

        # Second convolutional layer with 32 input channels and 64 output channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5) 

        # First fully connected layer, flattening the output of conv2 to 1024 and connecting to 128 nodes
        self.fc1 = nn.Linear(1024, 128)

        # Second fully connected layer that outputs to 10 classes
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Apply first convolution, followed by ReLU activation function and max pooling
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # Apply second convolution, followed by ReLU activation function and max pooling
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        # Flatten the output for the fully connected layer
        x = x.view(-1, 1024)

        # Apply first fully connected layer with ReLU activation function
        x = F.relu(self.fc1(x))
        
        # Apply second fully connected layer and compute log_softmax
        return F.log_softmax(self.fc2(x), dim=1)
