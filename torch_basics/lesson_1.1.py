import torch

"""
z = b + xw
"""
b = 0.1

x = [1.2, 2.2]

w = [3.3, 4.3]

# Tensor conversion below

b_t = torch.tensor(b)

x_t = torch.tensor(x)

w_t = torch.tensor(w)

z = x_t.dot(w_t) + b_t

print(f"z ouptut = {z}\n")
print(f"z type is {type(z)}")
print(f"Data type of z is {z.dtype}")