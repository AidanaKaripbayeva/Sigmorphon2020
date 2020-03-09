import numpy as np

import irisdata

X,Y_onehot = irisdata.load_iris_data()


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset


torch.manual_seed(1)


#
# The data needs to be as tensors, not numpy arrays.
#
X_ten = torch.tensor(X)
Y_ten = torch.tensor(Y_onehot,dtype=torch.float32)

#
#Define a loss function, it's just a sclar valued function on Tensors
#
loss_func = lambda X,Y: ((X-Y)**2).sum()

model = nn.Sequential(
	nn.Linear(4,5), #linear with 4 inputs and 5 outputs
	nn.Sigmoid(), #As an object instead of function for Sequential to hold.
	nn.Linear(5,3), #5 inputs and 3 outputs.
	nn.Sigmoid(), #another sigmoid.
)

#And optimizer for our model, the model automagicaly knows what its parameters are.
opt = torch.optim.SGD(model.parameters(),lr=0.1)

train_dataset = TensorDataset(X_ten, Y_ten) #keeps X and Y together and lets us slice together.
train_dataloader = DataLoader(train_dataset,batch_size=15,shuffle=True) #automatically handles batching and shuffling


#for 1000 epochs of training
for i in range(1000):
	#Each batch for SGD. The dataloader handles the batching.

	model.train() #Tell the model that we are training. Maybe sets some internal state.
	for xb, yb in train_dataloader:
		results = model(xb) #get results on the model all at once for the entire batch
		loss = loss_func(results, yb) #calculate the total loss on the batch.
		loss /= xb.shape[0] #loss as loss per item, not total, so that the realtive magnitude of the gradient isn't dependent on batch size.
							#So, Mean Squared Error
		loss.backward() #does back-propagation for the loss, doing it for all items at once.

		opt.step() #One step of the optimizer, adjusts parameters.
		opt.zero_grad() #Zeros out the gradient for the next batch?


	#Periodic output to the terminal.
	if i % 10 == 0:
		model.eval() #Tell the model we are doing model evaluation. Sets some internal state.
		xeval, yeval = train_dataset[:] #just using the full dataset as validation
		full_loss = loss_func(model(xeval),yeval) / xeval.shape[0] #MSE for dataset.
		print("Epoch ", i, "Loss", float(full_loss))

#Tell the model that we are evaluating it (not training)
model.eval()

print("Pred")
print( model(X_ten).detach().numpy().argmax(1) )

print("Ground Truth")
print( Y_onehot.argmax(1) )
