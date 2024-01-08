import numpy as np

def conv_3D_f(I3D, W3D, B3D, padding=True):

	(I_SIZE, _, I_DEPTH) = I3D.shape
	(K_SIZE, _, KI_DEPTH, KO_DEPTH) = W3D.shape
	(BO_DEPTH, ) = B3D.shape

	assert I_DEPTH == KI_DEPTH, "Input depth must be the same as weight depth"
	assert KO_DEPTH == BO_DEPTH, "Weight depth must be the same as bias depth"

	O_DEPTH = KO_DEPTH

	I3P = I3D
	if padding:
		I3P = np.pad(I3D, ((1,1), (1,1), (0,0)))

	O3D = np.zeros((I_SIZE,I_SIZE,O_DEPTH))

	for fn in range(O_DEPTH):
		for row in range(I_SIZE):
			for col in range(I_SIZE):
				Win3D = I3P[row:row+K_SIZE, col:col+K_SIZE]
				O3D[row,col,fn] = np.sum(Win3D*W3D[:,:,:,fn])+B3D[fn]

	return O3D

def max_pool_3D_f(I3D, P_SIZE=2):

	(I_SIZE, _, I_DEPTH) = I3D.shape
	H_SIZE = int(I_SIZE/2)

	P3D = np.zeros((H_SIZE,H_SIZE,I_DEPTH))

	for fn in range(I_DEPTH):
		for row in range(0,H_SIZE):
			for col in range(0,H_SIZE):
				Win3D = I3D[2*row:2*row+P_SIZE, 2*col:2*col+P_SIZE, fn]
				P3D[row,col,fn] = np.max(Win3D)

	return P3D

def max_pool_3D_b(O3Db, I3D, P_SIZE=2):

	(I_SIZE, _, I_DEPTH) = I3D.shape
	H_SIZE = int(I_SIZE/2)

	I3Db = np.zeros_like(I3D)

	for fn in range(I_DEPTH):
		for row in range(0,H_SIZE):
			for col in range(0,H_SIZE):
				Win3D = I3D[2*row:2*row+P_SIZE, 2*col:2*col+P_SIZE, fn]
				Win3Db = I3Db[2*row:2*row+P_SIZE, 2*col:2*col+P_SIZE, fn]
				Win3Db[Win3D==np.max(Win3D)] = O3Db[row,col,fn]

	return I3Db

def conv_3D_b(O3Db, I3D, W3D, B3D, padding=True, input_back=True):

	(I_SIZE, _, I_DEPTH) = I3D.shape
	(K_SIZE, _, KI_DEPTH, KO_DEPTH) = W3D.shape
	(BO_DEPTH,) = B3D.shape

	assert I_DEPTH == KI_DEPTH, "Input depth must be the same as weight depth"
	assert KO_DEPTH == BO_DEPTH, "Weight depth must be the same as bias depth"

	O_DEPTH = KO_DEPTH

	I3P = I3D
	if padding:
		I3P = np.pad(I3D, ((1,1), (1,1), (0,0)))

	W3Db = np.zeros_like(W3D)
	B3Db = np.zeros_like(B3D)

	for fn in range(O_DEPTH):
		for row in range(I_SIZE):
			for col in range(I_SIZE):
				Win3D = I3P[row:row+K_SIZE, col:col+K_SIZE]
				W3Db[:,:,:,fn] += Win3D*O3Db[row,col,fn]

	for fn in range(O_DEPTH):
		B3Db[fn] = np.sum(O3Db[:,:,fn])

	I3Db = None
	if input_back:
	
		I3Pb = np.zeros_like(I3P)

		for fn in range(O_DEPTH):
			for row in range(I_SIZE):
				for col in range(I_SIZE):
					Win3Db = I3Pb[row:row+K_SIZE, col:col+K_SIZE]
					Win3Db += O3Db[row,col,fn]*W3D[:,:,:,fn]

		I3Db = I3Pb
		if padding:
			I3Db = I3Pb[1:-1,1:-1,:]

	return W3Db, B3Db, I3Db