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
def load_iris_data(filename="./iris.data"):
	data = pn.read_csv(filename,names=["slen","swidth","plen","pwidth","class"])

	#read the data and turn it into a one-hot encoded output
	X,Y_names = np.hsplit(data.to_numpy(),[4])
	X = np.array(X,dtype=np.float32)

	name_map = dict(zip(np.unique(Y_names),count(0)))
	Y = np.array([name_map[i] for i in Y_names.ravel()]).reshape(-1,1)

	Y_onehot = np.zeros((Y.shape[0],len(name_map)))
	for i in range(Y.shape[0]):
		Y_onehot[i,Y[i,0]] = 1

	return X, Y_onehot
