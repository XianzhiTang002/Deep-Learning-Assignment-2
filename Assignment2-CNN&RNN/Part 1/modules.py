import numpy as np

class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Initializes a linear (fully connected) layer. 
        Initialize weights and biases.
        - Weights should be initialized to small random values (e.g., using a normal distribution).
        - Biases should be initialized to zeros.
        Formula: output = x * weight + bias
        """
        # Initialize weights and biases with the correct shapes.
        W = np.random.randn(in_features, out_features) / np.sqrt(in_features)
        b = np.zeros(out_features)
        self.params = {'weight': W, 'bias': b}
        self.grads = {'weight': None, 'bias': None}
        self.cache = None

    def forward(self, x):
        """
        Performs the forward pass using the formula: output = xW + b
        Implement the forward pass.
        """
        W = self.params['weight']
        b = self.params['bias']
        y = np.dot(x, W) + b
        self.cache = x
        return y

    def backward(self, dout):
        """
        Backward pass to calculate gradients of loss w.r.t. weights and inputs.
        Implement the backward pass.
        """
        x = self.cache
        W = self.params['weight']
        dW = np.dot(x.T, dout)
        db = np.sum(dout, axis=0)
        dx = np.dot(dout, W.T)
        self.grads['weight'] = dW
        self.grads['bias'] = db
        return dx

class ReLU(object):
    def forward(self, x):
        """
        Applies the ReLU activation function element-wise to the input.
        Formula: output = max(0, x)
        Implement the forward pass.
        """
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        """
        Computes the gradient of the ReLU function.
        Implement the backward pass.
        Hint: Gradient is 1 for x > 0, otherwise 0.
        """
        dx = dout * self.mask
        return dx

class SoftMax(object):
    def forward(self, x):
        """
        Applies the softmax function to the input to obtain output probabilities.
        Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j)) for all j
        Implement the forward pass using the Max Trick for numerical stability.
        """
        x_max = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - x_max)
        y = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return y

    def backward(self, dout):
        """
        The backward pass for softmax is often directly integrated with CrossEntropy for simplicity.
        Keep this in mind when implementing CrossEntropy's backward method.
        """
        # The implementation is in CrossEntropy. That is, when doing backward propagation, CrossEntropy acts as SoftmaxWithCrossEntropy
        return dout

class CrossEntropy(object):
    def forward(self, x, y):
        """
        Computes the CrossEntropy loss between predictions and true labels.
        Formula: L = -sum(y_i * log(p_i)), where p is the softmax probability of the correct class y.
        Implement the forward pass.
        """
        loss = -np.sum(y * np.log(x + 1e-10)) / x.shape[0]
        return loss

    def backward(self, x, y):
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        Implement the backward pass.
        Hint: For softmax output followed by cross-entropy loss, the gradient simplifies to: p - y.
        """
        N = x.shape[0]
        dx = (x - y) / N
        return dx
