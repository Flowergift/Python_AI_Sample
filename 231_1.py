x1, x2 = 2, 3
t = 27
w1 = 3
w2 = 4
b = 1

for epoch in range(2000):

	print('epoch = %d' %epoch)

	y = x1*w1 + x2*w2 + 1*b # ➊
	print(' y  = %6.3f' %y)

	E = (y-t)**2/2
	print(' E  = %.7f' %E)
	if E < 0.0000001:
		break

	yb = y - t
	x1b, x2b = yb*w1, yb*w2 # ➋
	w1b = yb*x1 # ➌
	w2b = yb*x2 # ➌
	bb = yb*1 # ➍
	print(' x1b, x2b = %6.3f, %6.3f'%(x1b, x2b))
	print(' w1b, w2b, bb = %6.3f, %6.3f, %6.3f'%(w1b, w2b, bb))

	lr = 0.01	
	w1 = w1 - lr*w1b # ➎ 
	w2 = w2 - lr*w2b # ➎
	b = b - lr*bb # ➏
	print(' w1,  w2,  b  = %6.3f, %6.3f, %6.3f'%(w1, w2, b))