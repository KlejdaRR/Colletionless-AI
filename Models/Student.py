import torch.nn as nn
import torch.nn.functional as F

class Student(nn.Module):
    def __init__(self):
        super(Student, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        # Fully connected layer 1
        # Input: 784 values (flattened image)
        # Output: 128 values
        # Function: Linear transformation: y = Wx + b
        # Weights: Matrix of size 128×784 + 128 biases
        # Purpose: First feature extraction from raw pixels

        self.fc2 = nn.Linear(128, 64)
        # Fully connected layer 2
        # Input: 128 values (from fc1)
        # Output: 64 values
        # Purpose: Further compression and feature abstraction

        self.fc3 = nn.Linear(64, 10)
        # Fully connected layer 3
        # Input: 64 values (from fc2)
        # Output: 10 values (one for each digit 0-9)
        # Purpose: Final classification layer

        self.dropout = nn.Dropout(0.2)
        # Dropout usage in order to prevent overfitting by forcing network to learn robust features

    def forward(self, x):
        # Step 1: Flattening image from [batch, 1, 28, 28] to [batch, 784]
        x = x.view(-1, 784) # squashing the image into a flat list

        # Step 2: fc1 + ReLU activation + Dropout
        x = F.relu(self.fc1(x))  # fc1: 784 → 128, then applying ReLU
        x = self.dropout(x)  # randomly disabling 20% of neurons

        # Step 3: fc2 + ReLU activation + Dropout
        x = F.relu(self.fc2(x))  # fc2: 128 → 64, then applying ReLU
        x = self.dropout(x)  # randomly disabling 20% of neurons

        # Step 4: fc3 (no activation yet)
        x = self.fc3(x)  # fc3: 64 → 10 (raw scores/logits)

        # Step 5: Converting to probabilities
        return F.log_softmax(x, dim=1)  # Output: log probabilities

    # INPUT IMAGE
    #     ↓
    # [28×28 = 784 pixels]
    #     ↓
    # FC1: 784 → 128
    #     ↓
    # ReLU Activation
    #     ↓
    # Dropout (20%)
    #     ↓
    # FC2: 128 → 64
    #     ↓
    # ReLU Activation
    #     ↓
    # Dropout (20%)
    #     ↓
    # FC3: 64 → 10
    #     ↓
    # LogSoftmax
    #     ↓
    # OUTPUT: 10 probabilities (digits 0-9)

    # In order to activate the usage of this neural network, the FORWARD function needs to be called.
    # If one feeds to the FORWARD function an input image (through its pixels), then this function firstly flattens
    # the image (because the image is fed through a 28x28 pixel format).
    # Then the FORWARD function applies a ReLu activation function to the output of the first fully connected layer
    # (fc1) of the Student Neural Network.
    # Then, it applies a random 20% dropout to the output of the ReLu activation function in order to prevent overfitting.
    # Next, the FORWARD function applies a ReLu activation function to the output of the second fully connected layer
    # (fc2) of the Student Neural Network (that would be the output of the first dropout layer).
    # Lastly, it applies again a random dropout of 20% to the output of the ReLu of the fc2. Then it just feeds
    # this last dropout to the last fully connected layer (no activation function is applied here ->
    # to the output of this layer).
    # Finally, it applies a logsoftmax activation function (gives logarithm of probabilities)
    # to the output of the third fully connected layer.