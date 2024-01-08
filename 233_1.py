x1, x2, x3 = 2, 3, 4
t1, t2, t3 = 27, -30, 179
w1, w4, w7 = 3, 5, 8
w2, w5, w8 = 4, 6, 9
w3, w6, w9 = 5, 7, 10
b1, b2, b3 = 1, 2, 3

for epoch in range(2000):

	print('epoch = %d' %epoch)

	y1 = x1*w1 + x2*w2 + x3*w3 + 1*b1 # ➊
	y2 = x1*w4 + x2*w5 + x3*w6 + 1*b2 # ➊
	y3 = x1*w7 + x2*w8 + x3*w9 + 1*b3 # ➊
	print(' y1,  y2,  y3  = %6.3f, %6.3f, %6.3f' %(y1, y2, y3))
		
	E = (y1-t1)**2/2 + (y2-t2)**2/2+ (y3-t3)**2/2
	print(' E  = %.7f' %E)
	if E < 0.0000001:
		break

	y1b, y2b, y3b = y1 - t1, y2 - t2, y3 - t3
	x1b = y1b*w1+y2b*w4+y3b*w7 # ➋
	x2b = y1b*w2+y2b*w5+y3b*w8 # ➋
	x3b = y1b*w3+y2b*w6+y3b*w9 # ➋
	w1b, w4b, w7b = x1*y1b, x1*y2b, x1*y3b # ➌
	w2b, w5b, w8b = x2*y1b, x2*y2b, x2*y3b # ➌
	w3b, w6b, w9b = x3*y1b, x3*y2b, x3*y3b # ➌
	b1b, b2b, b3b = 1*y1b, 1*y2b, 1*y3b # ➍
	print(' x1b, x2b, x2b = %6.3f, %6.3f, %6.3f'%(x1b, x2b, x3b))
	print(' w1b, w4b, w7b = %6.3f, %6.3f, %6.3f'%(w1b, w4b, w7b))
	print(' w2b, w5b, w8b = %6.3f, %6.3f, %6.3f'%(w2b, w5b, w8b))
	print(' w3b, w6b, w9b = %6.3f, %6.3f, %6.3f'%(w3b, w6b, w9b))
	print(' b1b, b2b, b3b = %6.3f, %6.3f, %6.3f'%(b1b, b2b, b3b))

	lr = 0.01 	
	w1, w4, w7 = w1 - lr*w1b, w4 - lr*w4b, w7 - lr*w7b # ➎
	w2, w5, w8 = w2 - lr*w2b, w5 - lr*w5b, w8 - lr*w8b # ➎
	w3, w6, w9 = w3 - lr*w3b, w6 - lr*w6b, w9 - lr*w9b # ➎
	b1, b2, b3 = b1 - lr*b1b, b2 - lr*b2b, b3 - lr*b3b # ➏
	print(' w1,  w4,  w7  = %6.3f, %6.3f, %6.3f'%(w1, w4, w7))
	print(' w2,  w5,  w8  = %6.3f, %6.3f, %6.3f'%(w2, w5, w8))
	print(' w3,  w6,  w9  = %6.3f, %6.3f, %6.3f'%(w3, w6, w9))
	print(' b1,  b2,  b3  = %6.3f, %6.3f, %6.3f'%(b1, b2, b3))