from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from pytorch_mlp import MLP
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

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
    targets = np.argmax(targets, axis=1)
    correct = np.sum(predictions == targets)
    accuracy = correct / len(targets)
    return accuracy

def train(X_train, X_test, y_train, y_test, dnn_hidden_units, learning_rate, max_steps, eval_freq, batch_size=4000, dataset="make_moons"):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    n_classes = y_train.shape[1]
    n_hidden = [int(x) for x in dnn_hidden_units.split(',') if x]

    model = MLP(n_inputs=input_dim, n_hidden=n_hidden, n_classes=n_classes)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr = learning_rate)
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(np.argmax(y_train, axis=1)).to(device)
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(np.argmax(y_test, axis=1)).to(device)

    max_iters = X_train.shape[0] // batch_size
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(max_steps):
        model.train()
        idx = torch.randperm(X_train_tensor.size(0))
        X_train_tensor = X_train_tensor[idx]
        y_train_tensor = y_train_tensor[idx]
        for iters in range(max_iters):
            start_idx = iters * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_train_tensor[start_idx:end_idx]
            y_batch = y_train_tensor[start_idx:end_idx]
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        if epoch % eval_freq == 0 or epoch == max_steps - 1:
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train_tensor)
                train_loss = criterion(train_outputs, y_train_tensor).item()
                train_accuracy = accuracy(train_outputs.detach().cpu().numpy(), y_train)
                train_losses.append(train_loss)
                train_accs.append(train_accuracy)

                test_outputs = model(X_test_tensor)
                test_loss = criterion(test_outputs, y_test_tensor).item()
                test_accuracy = accuracy(test_outputs.detach().cpu().numpy(), y_test)
                test_losses.append(test_loss)
                test_accs.append(test_accuracy)
            print(f'Epoch {epoch}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy * 100}%, Test Loss: {test_loss}, Test Accuracy: {test_accuracy * 100}%')

    return train_losses, train_accs, test_losses, test_accs


def main():
    """
    Main function
    """
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()