import torch

class Perceptron:
    def __init__(self, num_features):
        self._num_features = num_features
        self.weights = torch.zeros(self._num_features)
        self.bias = torch.tensor(0.0)

    def forward(self, x):
        weighted_sum = x.dot(self.weights) + self.bias
        
        if weighted_sum > 0:
            pred = torch.tensor(1.)
        else: 
            pred = torch.tensor(0.)
        return pred
    
    def update(self, x, true_y):
        prediction = self.forward(x)
        error = true_y - prediction

        self.bias += error
        self.weights += error * x

        return error

    def train(self, x_train, y_train, epochs):
        for ep in range(epochs):
            error_count = 0
            for x, y in zip(x_train, y_train):
                error = self.update(x, y)
                error_count += abs(error)
            print(f"Epoch : {ep + 1} errors {error_count}")
    
