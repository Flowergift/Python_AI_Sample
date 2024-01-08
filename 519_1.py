import numpy as np

np.random.seed(1)

image = np.random.randint(5, size=(4,4,2))
print('image_0 =\n', image[:,:,0])
print('image_1 =\n', image[:,:,1])

filter = np.random.randint(5, size=(3,3,2))
print('filter_0 =\n', filter[:,:,0])
print('filter_1 =\n', filter[:,:,1])

image_pad = np.pad(image,((1,1), (1,1), (0,0)))
print('image_pad_0 =\n', image_pad[:,:,0])
print('image_pad_1 =\n', image_pad[:,:,1])

convolution = np.zeros((4,4))

for row in range(4):
	for col in range(4):
		window = image_pad[row:row+3, col:col+3]
		convolution[row, col] = np.sum(window*filter)

print('convolution =\n', convolution)