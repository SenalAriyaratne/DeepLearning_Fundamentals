import numpy as np
import matplotlib.pyplot as plt

class DataGen:

    def __init__(self, num_points: int):
        self._num_data  = num_points
        self._num_features = 2
        self._num_classes = 2
        self._means = np.array([[0, 0], [3, 0.5]])
        self._std = np.array([[1, 1], [1, 1]])
        self.X = np.zeros((self._num_data, self._num_features))
        self.Y = np.zeros(self._num_data)
    
    def generate(self):
        for i in range(self._num_data):
            class_label = np.random.randint(0, self._num_classes)
            self.X[i] = np.random.normal(self._means[class_label], self._std[class_label], size=self._num_features)
            self.Y[i] = class_label
        return self.X, self.Y
    
    def plotter(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y)
        plt.xlabel('Feature 1 (x1)')
        plt.ylabel('Feature 2 (x2)')
        plt.title('Generated Data')
        plt.show()
    
