import torch
"""
lesson is about bradcasting
In Torch when we are computing with unequal tensor shapes-
- torch handles it implicitly by using methodology called broadcasting.

"""

x = torch.tensor([1.1,2.1,3.1,4.1])
y = torch.tensor([5.6])
z = x + y
print(f"Result with bradcasting in action  = {z}")
print("\n\n")

a = torch.tensor([[1.1,2.1,3.1,4.1],
                 [2.2,3.2,4.2,5.2]])
b = torch.tensor([3.3, 4.3, 5.3 ,6.3])

c = a + b
print(f"Result with broadcasting in action = {c}")