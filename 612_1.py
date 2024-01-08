import numpy as np

np.set_printoptions(formatter={'float_kind':lambda x: "{0:6.3f}".format(x)})

IMG_SIZE = 4
HALF_SIZE = int(IMG_SIZE/2)

I_DEPTH = 1
C_DEPTH = 2
K_SIZE = 3
P_SIZE = 2
NUM_OUT = 3

np.random.seed(1)

I = np.random.rand(1,IMG_SIZE,IMG_SIZE,I_DEPTH)
WC = np.random.rand(K_SIZE,K_SIZE,I_DEPTH,C_DEPTH)
BC = np.random.rand(C_DEPTH,)

WO = np.random.rand(HALF_SIZE*HALF_SIZE*C_DEPTH,NUM_OUT)
BO = np.random.rand(1, NUM_OUT)

T = np.zeros((1, NUM_OUT))

C = np.zeros((IMG_SIZE,IMG_SIZE,C_DEPTH))
P = np.zeros((HALF_SIZE,HALF_SIZE,C_DEPTH))

IP = np.pad(I, ((0,0), (1,1), (1,1), (0,0)))
print('IP  =\n', IP[0,:,:,0])