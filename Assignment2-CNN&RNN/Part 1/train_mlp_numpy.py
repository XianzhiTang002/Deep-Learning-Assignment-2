import argparse
import numpy as np
from mlp_numpy import MLP  
from modules import CrossEntropy
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500 # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.
    
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding
    
    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    # Implement the accuracy calculation
    # Hint: Use np.argmax to find predicted classes, and compare with the true classes in targets
    pred = np.argmax(predictions, axis=1)
    label = np.argmax(targets, axis=1)
    acc = np.mean(pred == label)
    return acc

def train(X_train, X_test, y_train, y_test, dnn_hidden_units, learning_rate, max_steps, eval_freq, is_batch=True, batch_size=4000, dataset="make_moons"):
    """
    Performs training and evaluation of MLP model.
    
    Args:
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        NOTE: Add necessary arguments such as the data, your model...
    """
    n_classes = y_train.shape[1]
    
    # Initialize your MLP model and loss function (CrossEntropy) here
    n_hidden = [int(u) for u in dnn_hidden_units.split(',')]
    mlp = MLP(X_train.shape[1], n_hidden, n_classes)
    loss_layer = CrossEntropy()
    
    # Use mini-batch
    if not is_batch:
        batch_size = 1
    max_iters = X_train.shape[0] // batch_size
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    for step in range(max_steps):
        # Implement the training loop
        # 1. Forward pass
        # 2. Compute loss
        # 3. Backward pass (compute gradients)
        # 4. Update weights
        
        # Shuffle
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]
        for iters in range(max_iters):
            batch_X = X_train[iters * batch_size:(iters + 1) * batch_size]
            batch_y = y_train[iters * batch_size:(iters + 1) * batch_size]
            
            y_pred = mlp.forward(batch_X)
            loss = loss_layer.forward(y_pred, batch_y)
            dloss = loss_layer.backward(y_pred, batch_y)
            mlp.backward(dloss)
            mlp.update(learning_rate)
        if step % eval_freq == 0 or step == max_steps - 1:
            # Evaluate the model on the test set
            # 1. Forward pass on the test set
            # 2. Compute loss and accuracy
            y_pred_train = mlp.forward(X_train)
            y_pred_test = mlp.forward(X_test)
            test_loss = loss_layer.forward(y_pred_test, y_test)
            train_acc = accuracy(y_pred_train, y_train)
            test_acc = accuracy(y_pred_test, y_test)
            train_losses.append(loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            print(f"Step: {step}, Loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%")
    
    print("Training complete!")
    return train_losses, train_accs, test_losses, test_accs

def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS = parser.parse_known_args()[0]
    
    train(FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq)

if __name__ == '__main__':
    main()
