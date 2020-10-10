import torch
import torch.nn as nn

class Net(nn.Module):   
    def __init__(self):
        super(Net, self).__init__()
        self.leakyRelu = nn.LeakyReLU(negative_slope = 1, inplace=True)

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(16, 4, kernel_size=5),
            self.leakyRelu,
            # Defining another 2D convolution layer
            nn.Conv2d(4, 1, kernel_size=5),
            self.leakyRelu,
            nn.Flatten()
        )
        # self.temp = len(self.cnn_layers)
        self.linear_layers = nn.Sequential(
            nn.Linear(5632, 19),
            nn.Softmax()
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x.float())
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.linear_layers(x)
        return x