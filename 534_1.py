import numpy as np

np.random.seed(1)

X = np.random.randint(5, size=(3,3))
print('X =\n', X)

W = np.random.randint(5, size=(3,3))
print('W =\n', W)

b = 0
print('b =', b)

y = np.sum(X*W) + b
print('y =', y)

yb = np.random.randint(-5,5)
print('yb =', yb)

Wb = yb*X
print('Wb =\n', Wb)

bb = yb
print('bb =', bb)

Xb = yb*W
print('Xb =\n', Xb)