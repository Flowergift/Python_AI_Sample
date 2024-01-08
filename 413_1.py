import numpy as np

np.set_printoptions(formatter={'float_kind':lambda x: "{0:6.3f}".format(x)})
 
X = np.array([[2, 3]])
T = np.array([[27]])

W = np.array([[3],
              [4]])
B = np.array([[1]])

for epoch in range(2):

	print('epoch = %d' %epoch)

	Y = X @ W + B # ➊
	print(' Y  =', Y)

	E = np.sum((Y - T) ** 2) / Y.shape[1] # ➎

	Yb = 2 * (Y - T) / Y.shape[1] # ➏

	Wb = X.T @ Yb # ➌
	Bb = 1 * Yb # ➍

	lr = 0.01

	W = W - lr * Wb # ➐
	B = B - lr * Bb # ➑
	print(' W  =\n', W)
	print(' B  =', B)