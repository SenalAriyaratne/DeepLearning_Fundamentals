import torch

"""
matmul using torch
"""
b = torch.tensor([0.])

X = torch.tensor([[1.2,2.2],
                  [4.4,5.5]])

w = torch.tensor([3.3,4.3])

z = X.matmul(w) + b

print(f" z output after matrix multiplication = {z}")
print("---------------------------------------------")
print("\n\n")

"""
matmul of two matrices
1.To do this task first we need to transpose the weight matrix
"""

X_mat = torch.rand(100,10)

W_mat = torch.rand(50,10)

W_mat_T = W_mat.T

result = torch.matmul(X_mat, W_mat_T)

print(f"Shape of Weight matrix before transpose = {W_mat.shape}")
print(f"Shape Weight matrix after transpose = {W_mat_T.shape}")
print(f"Shape of Training data matrix X = {X_mat.shape}")
print(f"Shape of result matrix = {result.shape}")