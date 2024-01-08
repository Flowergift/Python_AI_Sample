import numpy as np

np.random.seed(1)

X = np.random.randint(5, size=(4,4))
print('X =\n', X)

W = np.random.randint(5, size=(3,3,2)) #
print('W_0 =\n', W[:,:,0]) #
print('W_1 =\n', W[:,:,1]) #

b = np.zeros((2,)) #
print('b =', b)

XP = np.pad(X,((1,1), (1,1)))
print('XP =\n', XP)

Y = np.zeros((4,4,2)) #
 
for fn in range(2): #
	for row in range(4):
		for col in range(4):
			winXP = XP[row:row+3, col:col+3]
			Y[row, col, fn] = np.sum(winXP*W[:,:,fn]) + b[fn] #

print('Y_0 =\n', Y[:,:,0]) #
print('Y_1 =\n', Y[:,:,1]) #

MP = np.zeros((2,2,2)) #
 
for fn in range(2): #
	for row in range(0, 2):
		for col in range(0, 2):
			winY = Y[2*row:2*row+2, 2*col:2*col+2, fn] #
			MP[row, col, fn] = np.max(winY) #

print('MP_0 =\n', MP[:,:,0]) #
print('MP_1 =\n', MP[:,:,1]) #

MPb = np.random.randint(-8,5, size=MP.shape)
print('MPb_0 =\n', MPb[:,:,0]) #
print('MPb_1 =\n', MPb[:,:,1]) #

Yb = np.zeros_like(Y)

for fn in range(2): #
	for row in range(0, 2):
		for col in range(0, 2):
			winY = Y[2*row:2*row+2, 2*col:2*col+2, fn] #
			winYb = Yb[2*row:2*row+2, 2*col:2*col+2, fn] #
			winYb[winY==np.max(winY)]=MPb[row, col, fn] #

print('Yb_0 =\n', Yb[:,:,0]) #
print('Yb_1 =\n', Yb[:,:,1]) #

Wb = np.zeros_like(W)/1

for fn in range(2): #
	for row in range(4):
		for col in range(4):
			winXP = XP[row:row+3, col:col+3]
			Wb[:,:,fn] += winXP*Yb[row, col, fn] #

print('Wb_0 =\n', Wb[:,:,0]) #
print('Wb_1 =\n', Wb[:,:,1]) #

bb = np.zeros_like(b)/1 #

for fn in range(2): #
	bb[fn] = np.sum(Yb[:,:,fn]) #

print('bb =', bb)

XPb = np.zeros_like(XP)/1

for fn in range(2): #
	for row in range(4):
		for col in range(4):
			winXPb = XPb[row:row+3, col:col+3]
			winXPb += Yb[row, col, fn]*W[:,:,fn] #

print('XPb =\n', XPb)

Xb = XPb[1:-1,1:-1]
print('Xb =\n', Xb)