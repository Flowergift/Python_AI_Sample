import numpy as np

np.random.seed(1)

X = np.random.randint(5, size=(4,4)) #
print('X =\n', X)

W = np.random.randint(5, size=(3,3))
print('W =\n', W)

b = 0
print('b =', b)

Y = np.zeros((2,2)) #

for row in range(2): #
	for col in range(2): #
		winX = X[row:row+3, col:col+3] #
		Y[row, col] = np.sum(winX*W) + b #
        
print('Y =\n', Y) #

Yb = np.random.randint(-8,5, size=Y.shape) #
print('Yb =\n', Yb) #

Wb = np.zeros_like(W) #

for row in range(2): #
	for col in range(2): #
		winX = X[row:row+3, col:col+3] #
		# print('winX(%d, %d) =\n' %(row, col), winX) #
		Wb += winX*Yb[row, col] #
		# print('Wb(%d, %d) =\n' %(row, col), Wb) #

print('Wb =\n', Wb)

bb = np.sum(Yb) #
print('bb =', bb)

Xb = np.zeros_like(X) #

for row in range(2): #
	for col in range(2): #
		winXb = Xb[row:row+3, col:col+3] #
		winXb += Yb[row, col]*W #
		# print('Xb(%d, %d) =\n' %(row, col), Xb) #

print('Xb =\n', Xb)