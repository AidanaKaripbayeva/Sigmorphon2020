
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
