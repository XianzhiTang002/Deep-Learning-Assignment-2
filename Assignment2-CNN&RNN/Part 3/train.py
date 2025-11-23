from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN

def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model that we are going to use
    model = VanillaRNN(
        seq_length=config.input_length,
        input_dim=config.input_dim,
        hidden_dim=config.num_hidden,
        output_dim=config.num_classes,
        batch_size=config.batch_size
    )
    model.to(device)

    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=config.learning_rate)
    accuracies = []
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):

        # Add more code here ...
        batch_inputs = batch_inputs.unsqueeze(-1).float().to(device)
        batch_targets = batch_targets.long().to(device)
        optimizer.zero_grad()
        outputs = model(batch_inputs)
        log_output = torch.log(outputs + 1e-8)
        loss = criterion(log_output, batch_targets)
        loss.backward()

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
        optimizer.step()

        # Add more code here ...

        loss = loss.item()
        _, predicted = torch.max(outputs, 1)
        total = batch_targets.size(0)
        correct = (predicted == batch_targets).sum().item()
        accuracy = correct / total
        accuracies.append(accuracy)

        if step % 10 == 0:
            # print acuracy/loss here
            print(f'Length: {config.input_length}, Step: {step}, Loss: {loss:.4f}, Accuracy: {accuracy * 100:.4f}%')

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    print('Done training.')
    return accuracies

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=10000, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    config = parser.parse_args()
    # Train the model
    train(config)