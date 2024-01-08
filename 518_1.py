import numpy as np

np.random.seed(1)

image = np.random.randint(5, size=(3,3,2))
print('image_0 =\n', image[:,:,0])
print('image_1 =\n', image[:,:,1])

filter = np.random.randint(5, size=(3,3,2))
print('filter_0 =\n', filter[:,:,0])
print('filter_1 =\n', filter[:,:,1])

image_x_filter = image * filter
print('image_x_filter_0 =\n', image_x_filter[:,:,0])
print('image_x_filter_1 =\n', image_x_filter[:,:,1])

convolution = np.sum(image_x_filter)
print('convolution =\n', convolution)