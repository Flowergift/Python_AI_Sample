x1, x2 = 2, 3
t1, t2 = 27, -30
w1, w3 = 3, 5
w2, w4 = 4, 6
b1, b2 = 1, 2

for epoch in range(2000):

	print('epoch = %d' %epoch)

	y1 = x1*w1 + x2*w2 + 1*b1 # ➊
	y2 = x1*w3 + x2*w4 + 1*b2 # ➊
	print(' y1,  y2  = %6.3f, %6.3f' %(y1, y2))

	E = (y1-t1)**2/2 + (y2-t2)**2/2
	print(' E  = %.7f' %E)
	if E < 0.0000001:
		break

	y1b, y2b = y1 - t1, y2 - t2 
	x1b, x2b = y1b*w1+y2b*w3, y1b*w2+y2b*w4 # ➋
	w1b, w3b = x1*y1b, x1*y2b # ➌
	w2b, w4b = x2*y1b, x2*y2b # ➌
	b1b, b2b = 1*y1b, 1*y2b # ➍
	print(' x1b, x2b = %6.3f, %6.3f'%(x1b, x2b))
	print(' w1b, w3b = %6.3f, %6.3f'%(w1b, w3b))
	print(' w2b, w4b = %6.3f, %6.3f'%(w2b, w4b))
	print(' b1b, b2b = %6.3f, %6.3f'%(b1b, b2b))

	lr = 0.01
	w1, w3 = w1 - lr*w1b, w3 - lr*w3b # ➎
	w2, w4 = w2 - lr*w2b, w4 - lr*w4b # ➎
	b1, b2 = b1 - lr*b1b, b2 - lr*b2b # ➏
	print(' w1,  w3  = %6.3f, %6.3f'%(w1, w3))
	print(' w2,  w4  = %6.3f, %6.3f'%(w2, w4))
	print(' b1,  b2  = %6.3f, %6.3f'%(b1, b2))