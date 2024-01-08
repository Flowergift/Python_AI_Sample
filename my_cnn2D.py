import numpy as np

def conv_2D_f(I3D, W3D, B3D, padding=True):

	(I_SIZE, _, I_DEPTH) = I3D.shape
	(K_SIZE, _, KI_DEPTH, KO_DEPTH) = W3D.shape
	(BO_DEPTH, ) = B3D.shape

	assert I_DEPTH == KI_DEPTH, "Input depth must be the same as weight depth"
	assert KO_DEPTH == BO_DEPTH, "Weight depth must be the same as bias depth"

	O_DEPTH = KO_DEPTH

	I3P = I3D
	if padding:
		I3P = np.pad(I3D, ((1,1), (1,1), (0,0)))

	I3S = np.zeros((I_SIZE, I_SIZE, K_SIZE, K_SIZE, I_DEPTH))

	for row in range(I_SIZE):
		for col in range(I_SIZE):
			Win3D = I3P[row:row+K_SIZE, col:col+K_SIZE]
			I3S[row,col] = Win3D

	I3R = I3S.reshape(I_SIZE*I_SIZE, K_SIZE, K_SIZE, I_DEPTH)

	I2D = I3R.reshape(1*I_SIZE*I_SIZE, K_SIZE*K_SIZE*I_DEPTH)
	W2D = W3D.reshape(K_SIZE*K_SIZE*I_DEPTH,O_DEPTH)
	B2D = B3D

	O2D = I2D@W2D + B2D

	O3D = O2D.reshape(I_SIZE, I_SIZE, O_DEPTH)

	cache = (I3R, I3S, I3P)

	return O3D, cache

def conv_2D_b(O3Db, I3D, W3D, B3D, cache, padding=True, input_back=True):

	(I_SIZE, _, I_DEPTH) = I3D.shape
	(K_SIZE, _, KI_DEPTH, KO_DEPTH) = W3D.shape
	(BO_DEPTH, ) = B3D.shape

	assert I_DEPTH == KI_DEPTH, "Input depth must be the same as weight depth"
	assert KO_DEPTH == BO_DEPTH, "Weight depth must be the same as bias depth"

	O_DEPTH = KO_DEPTH

	(I3R, I3S, I3P) = cache

	I2D = I3R.reshape(1*I_SIZE*I_SIZE, K_SIZE*K_SIZE*I_DEPTH)
	W2D = W3D.reshape(K_SIZE*K_SIZE*I_DEPTH,O_DEPTH)

	O2Db = O3Db.reshape(I_SIZE*I_SIZE,O_DEPTH)

	W2Db = I2D.T@O2Db

	B2Db = np.zeros_like(B3D)
	for fn in range(O_DEPTH):
		B2Db[fn] = np.sum(O2Db[:,fn])

	W3Db = W2Db.reshape(W3D.shape)
	B3Db = B2Db.reshape(B3D.shape)

	I3Db = None
	if input_back:

		I2Db = O2Db@W2D.T

		I3Rb = I2Db.reshape(I3R.shape)
		I3Sb = I3Rb.reshape(I3S.shape)
		I3Pb = np.zeros_like(I3P)

		for row in range(I_SIZE):
			for col in range(I_SIZE):
				Win3Db = I3Sb[row,col]
				I3Pb[row:row+K_SIZE, col:col+K_SIZE] += Win3Db

		I3Db = I3Pb
		if padding:
			I3Db = I3Pb[1:-1,1:-1,:]

	return W3Db, B3Db, I3Db