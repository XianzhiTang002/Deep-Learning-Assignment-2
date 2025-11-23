import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(256, 10)
        self.network = nn.Sequential(
            self.fc1,
            nn.ReLU(),
            self.dropout1,
            self.fc2,
            nn.ReLU(),
            self.dropout2,
            self.fc3,
            self.dropout3,
            self.fc4
        )

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)  # Flatten the input
        return self.network(x)

def train(model, train_loader, test_loader, criterion, optimizer, epochs, eval_freq):
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    for epoch in range(epochs):
        model.train()
        train_total_loss = 0
        train_correct = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
        if epoch % eval_freq == 0 or epoch == epochs - 1:
            train_loss = train_total_loss / len(train_loader)
            train_accuracy = train_correct / len(train_loader.dataset)
            train_losses.append(train_loss)
            train_accs.append(train_accuracy)

            model.eval()
            test_total_loss = 0
            test_correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = model(data)
                    loss = criterion(output, target)
                    test_total_loss += loss.item()
                    pred = output.argmax(dim=1, keepdim=True)
                    test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss = test_total_loss / len(test_loader)
            test_accuracy = test_correct / len(test_loader.dataset)
            test_losses.append(test_loss)
            test_accs.append(test_accuracy)
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    return train_losses, train_accs, test_losses, test_accs