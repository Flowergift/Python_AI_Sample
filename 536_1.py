import numpy as np

np.random.seed(1)

X = np.random.randint(5, size=(4,4))
print('X =\n', X)

W = np.random.randint(5, size=(3,3))
print('W =\n', W)

b = 0
print('b =', b)

XP = np.pad(X,((1,1), (1,1))) #
print('XP =\n', XP) #

Y = np.zeros((4,4)) #

for row in range(4): #
	for col in range(4): #
		winXP = XP[row:row+3, col:col+3] #
		Y[row, col] = np.sum(winXP*W) + b #
        
print('Y =\n', Y) 

Yb = np.random.randint(-8,5, size=Y.shape) 
print('Yb =\n', Yb) 

Wb = np.zeros_like(W) 

for row in range(4): #
	for col in range(4): #
		winXP = XP[row:row+3, col:col+3] #
		Wb += winXP*Yb[row, col] #
		# print('Wb(%d, %d) =\n' %(row, col), Wb)

print('Wb =\n', Wb)

bb = np.sum(Yb) 
print('bb =', bb)

XPb = np.zeros_like(XP) #

for row in range(4): #
	for col in range(4): #
		winXPb = XPb[row:row+3, col:col+3] #
		winXPb += Yb[row, col]*W #
		# print('XPb(%d, %d) =\n' %(row, col), XPb)

print('XPb =\n', XPb) #

Xb = XPb[1:-1,1:-1] #
print('Xb =\n', Xb)