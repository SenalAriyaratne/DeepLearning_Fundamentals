import torch
from data_loader.datagen import DataGen
from model.perceptron import Perceptron

q = DataGen(40)
x, y = q.generate()
xt = torch.tensor(x, dtype=torch.float32)
yt = torch.tensor(y, dtype=torch.long)
#q.plotter()
#print(torch.tensor(x, dtype=torch.float32))
# print(xt[2])
# print(yt[2])

# ppn = Perceptron(2)

# out = ppn.update(xt[2], yt[2])
# print(out)
# print("Model parapmeteers")
# print(f"Weights : {ppn.weights}")
# print(f"Bias : {ppn.bias}")

model = Perceptron(2)

model.train(xt,yt, 10)


