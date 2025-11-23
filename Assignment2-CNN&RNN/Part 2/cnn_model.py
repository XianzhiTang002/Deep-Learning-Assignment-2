from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn

class CNN(nn.Module):

  def __init__(self, n_channels, n_classes):
    """
    Initializes CNN object. 
    
    Args:
      n_channels: number of input channels
      n_classes: number of classes of the classification problem
    """
    super(CNN, self).__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )
    self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.conv2 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU()
    )
    self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.conv3 = nn.Sequential(
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU()
    )
    self.conv4 = nn.Sequential(
        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU()
    )
    self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.conv5 = nn.Sequential(
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )
    self.conv6 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )
    self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.conv7 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )
    self.conv8 = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU()
    )
    self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    self.fc = nn.Linear(512, n_classes)

  def forward(self, x):
    """
    Performs forward pass of the input.
    
    Args:
      x: input to the network
    Returns:
      out: outputs of the network
    """
    out = self.conv1(x)
    out = self.pool1(out)
    out = self.conv2(out)
    out = self.pool2(out)
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.pool3(out)
    out = self.conv5(out)
    out = self.conv6(out)
    out = self.pool4(out)
    out = self.conv7(out)
    out = self.conv8(out)
    out = self.pool5(out)
    out = out.view(out.size(0), -1)
    out = self.fc(out)
    return out
