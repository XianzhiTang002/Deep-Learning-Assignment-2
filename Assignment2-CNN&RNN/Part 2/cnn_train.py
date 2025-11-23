from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from cnn_model import CNN
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = '../data'

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    predictions = np.argmax(predictions, axis=1)
    correct = np.sum(predictions == targets)
    accuracy = correct / len(targets)
    return accuracy

def train(learning_rate, max_steps, eval_freq, batch_size):
    """
    Performs training and evaluation of CNN model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = CNN(n_channels=3, n_classes=10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(max_steps):
        model.train()
        train_total_loss = 0
        train_predictions = []
        train_targets = []
        for batch_inputs, batch_targets in train_loader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()
            train_predictions.append(outputs.detach().cpu().numpy())
            train_targets.append(batch_targets.detach().cpu().numpy())

        if epoch % eval_freq == 0 or epoch == max_steps - 1:
            train_loss = train_total_loss / len(train_loader)
            train_losses.append(train_loss)
            train_predictions = np.concatenate(train_predictions, axis=0)
            train_targets = np.concatenate(train_targets, axis=0)
            train_accuracy = accuracy(train_predictions, train_targets)
            train_accs.append(train_accuracy)

            model.eval()
            test_predictions = []
            test_targets = []
            test_total_loss = 0
            with torch.no_grad():
                for batch_inputs, batch_targets in test_loader:
                    batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
                    outputs = model(batch_inputs)
                    loss = criterion(outputs, batch_targets)
                    test_total_loss += loss.item()
                    test_predictions.append(outputs.detach().cpu().numpy())
                    test_targets.append(batch_targets.detach().cpu().numpy())
            test_loss = test_total_loss / len(test_loader)
            test_losses.append(test_loss)
            test_predictions = np.concatenate(test_predictions, axis=0)
            test_targets = np.concatenate(test_targets, axis=0)
            test_accuracy = accuracy(test_predictions, test_targets)
            test_accs.append(test_accuracy)
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy * 100:.4f}%, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy * 100:.4f}%')

def main():
    """
    Main function
    """
    train()

if __name__ == '__main__':
  # Command line arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()