import torch

m = torch.tensor([[1., 2. ,3. ],
                  [4. ,5. , 6.]])

n = torch.tensor([[[1., 2. ,3. ],
                  [4. ,5. , 6.]]])

print(f"\nTensor Info ---- m ----")
print(f"1 - Tensor m is {m}")
print(f"2 - Shape of Tensor m is {m.shape}")
print(f"3 - The Rank / Number of Dimensions of Tensor m is {m.ndim}")
print(f"4 - In a two dimensional tensor : No. of rows = {m.shape[0]} and the No.of cols = {m.shape[1]}")

print(f"\nTensor Info ---- n ----")
print(f"1 - Tensor m is {n}")
print(f"2 - Shape of Tensor m is {n.shape}")
print(f"3 - The Rank / Number of Dimensions of Tensor m is {n.ndim}")