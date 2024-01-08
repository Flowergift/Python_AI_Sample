import numpy as np

np.set_printoptions(formatter={'float_kind':lambda x: "{0:6.3f}".format(x)})

X = np.array([[2, 3]])
T = np.array([[27, -30]])
W = np.array([[3, 5],
              [4, 6]])
B = np.array([[1, 2]])

for epoch in range(1000):

	print('epoch = %d' %epoch)

	Y = X @ W + B # ➊
	print(' Y  =', Y)

	E = np.sum((Y - T) ** 2 / 2)
	print(' E  = %.7f' %E)
	if E < 0.0000001:
		break

	Yb = Y - T
	Xb = Yb @ W.T # ➋
	Wb = X.T @ Yb # ➌
	Bb = 1 * Yb # ➍
	print(' Xb =\n', Xb)
	print(' Wb =\n', Wb)
	print(' Bb =\n', Bb)

	lr = 0.01
	W = W - lr * Wb # ➎
	B = B - lr * Bb # ➏
	print(' W  =\n', W)
	print(' B  =\n', B)