xs = [-1., 0., 1., 2.,  3.,  4.]
ys = [-2., 1., 4., 7., 10., 13.]
w = 10.
b = 10.

y = xs[0]*w + 1*b
print("x  = %6.3f, y  = %6.3f" %(xs[0], y))

t = ys[0]
E = (y-t)**2/2
print('E  = %.7f' %E)

yb = y - t
wb = yb*xs[0]
bb = yb*1
print('wb = %6.3f, bb = %6.3f'%(wb, bb))

lr = 0.01
w = w - lr*wb
b = b - lr*bb
print('w  = %6.3f, b  = %6.3f'%(w, b))	