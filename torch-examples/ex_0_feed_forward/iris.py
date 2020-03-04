import pandas as pn

_ = """
   1. sepal length in cm
   2. sepal width in cm
   3. petal length in cm
   4. petal width in cm
   5. class:
      -- Iris Setosa
      -- Iris Versicolour
      -- Iris Virginica
"""

import pandas as pn
import numpy as np
from itertools import count

data = pn.read_csv("./iris.data",names=["slen","swidth","plen","pwidth","class"])

#read the data and turn it into a one-hot encoded output
X,Y_names = np.hsplit(data.to_numpy(),[4])
X = np.array(X,dtype=np.float32)

name_map = dict(zip(np.unique(Y_names),count(0)))
Y = np.array([name_map[i] for i in Y_names.ravel()]).reshape(-1,1)

Y_onehot = np.zeros((Y.shape[0],len(name_map)))
for i in range(Y.shape[0]):
	Y_onehot[i,Y[i,0]] = 1


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(1)

class Iris_Two_Layer_NN(nn.Module):
	def __init__(self):
		super().__init__()
		self.a1 = nn.Linear(4,5) #linear with 4 inputs and 5 outputs
		self.s1 = nn.Sigmoid() #Not sure if I really need multiple sigmoids
		self.a2 = nn.Linear(5,3) #5 inputs and 3 outputs.
		self.s2 = nn.Sigmoid() #another sigmoid.

	def forward(self, xb):
		activation_one = self.a1(xb)
		nonlinear_one = self.s1(activation_one)

		activation_two = self.a2(nonlinear_one)
		nonlinear_two = self.s2(activation_two)

		return nonlinear_two

#
# The data needs to be as tensors, not numpy arrays.
#
X_ten = torch.tensor(X)
Y_ten = torch.tensor(Y_onehot,dtype=torch.float32)

#
#Define a loss function, it's just a sclar valued function on Tensors
#
loss_func = lambda X,Y: ((X-Y)**2).sum()

model = Iris_Two_Layer_NN()

#And optimizer for our model, the model automagicaly knows what its parameters are.
opt = torch.optim.SGD(model.parameters(),lr=0.001)

train_dataset = TensorDataset(X_ten, Y_ten)
train_dataloader = DataLoader(train_dataset,batch_size=15)

#Tell the model that we are training
model.train()

for i in range(1000):
	for xb, yb in train_dataloader:
		results = model(xb)
		loss = loss_func(results, yb)
		loss.backward()

		opt.step()
		opt.zero_grad()

print("Pred")
print( model(X_ten).detach().numpy().argmax(1) )

print("Ground Truth")
print( Y.ravel() )
