import numpy as np

np.random.seed(1)

X = np.random.randint(5, size=(4,4,2)) #
print('X_0 =\n', X[:,:,0]) #
print('X_1 =\n', X[:,:,1]) #

W = np.random.randint(5, size=(3,3,2,2)) #
print('W_00 =\n', W[:,:,0,0]) #
print('W_10 =\n', W[:,:,1,0]) #
print('W_01 =\n', W[:,:,0,1]) #
print('W_11 =\n', W[:,:,1,1]) #

b = np.zeros((2,))
print('b =', b)

XP = np.pad(X,((1,1), (1,1), (0,0))) #
print('XP_0 =\n', XP[:,:,0]) #
print('XP_1 =\n', XP[:,:,1]) #

Y = np.zeros((4,4,2))

for fn in range(2):
	for row in range(4):
		for col in range(4):
			winXP = XP[row:row+3, col:col+3]
			Y[row, col, fn] = np.sum(winXP*W[:,:,:,fn]) + b[fn] #

print('Y_0 =\n', Y[:,:,0])
print('Y_1 =\n', Y[:,:,1])

MP = np.zeros((2,2,2))

for fn in range(2):
	for row in range(0, 2):
		for col in range(0, 2):
			winY = Y[2*row:2*row+2, 2*col:2*col+2, fn]
			MP[row, col, fn] = np.max(winY)

print('MP_0 =\n', MP[:,:,0])
print('MP_0 =\n', MP[:,:,1])

MPb = np.random.randint(-8,5, size=MP.shape)
print('MPb_0 =\n', MPb[:,:,0])
print('MPb_0 =\n', MPb[:,:,1])

Yb = np.zeros_like(Y)

for fn in range(2):
	for row in range(0, 2):
		for col in range(0, 2):
			winY = Y[2*row:2*row+2, 2*col:2*col+2, fn]
			winYb = Yb[2*row:2*row+2, 2*col:2*col+2, fn]
			winYb[winY==np.max(winY)] = MPb[row, col, fn]

print('Yb_0 =\n', Yb[:,:,0])
print('Yb_1 =\n', Yb[:,:,1])

Wb = np.zeros_like(W)/1

for fn in range(2):
	for row in range(4):
		for col in range(4):
			winXP = XP[row:row+3, col:col+3]
			Wb[:,:,:,fn] += winXP*Yb[row, col, fn] #

print('Wb_00 =\n', Wb[:,:,0,0]) #
print('Wb_10 =\n', Wb[:,:,1,0]) #
print('Wb_01 =\n', Wb[:,:,0,1]) #
print('Wb_11 =\n', Wb[:,:,1,1]) #

bb = np.zeros_like(b)/1

for fn in range(2):
	bb[fn] = np.sum(Yb[:,:,fn])

print('bb =', bb)

XPb = np.zeros_like(XP)/1

for fn in range(2):
	for row in range(4):
		for col in range(4):
			winXPb = XPb[row:row+3, col:col+3]
			winXPb += Yb[row, col, fn]*W[:,:,:,fn] #

print('XPb_0 =\n', XPb[:,:,0]) #
print('XPb_1 =\n', XPb[:,:,1]) #

Xb = XPb[1:-1,1:-1]
print('Xb_0 =\n', Xb[:,:,0]) #
print('Xb_1 =\n', Xb[:,:,1]) #