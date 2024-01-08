i1, i2 = .05, .10
t1, t2 = .01, .99

w1, w3 = .15, .25
w2, w4 = .20, .30
b1, b2 = .35, .35

w5, w7 = .40, .50
w6, w8 = .45, .55
b3, b4 = .60, .60

for epoch in range(2000):

	print('epoch = %d' %epoch)

	h1 = i1*w1 + i2*w2 + 1*b1 # ➊
	h2 = i1*w3 + i2*w4 + 1*b2 # ➊
	o1 = h1*w5 + h2*w6 + 1*b3 # ➋
	o2 = h1*w7 + h2*w8 + 1*b4 # ➋
	print(' h1,  h2  = %6.3f, %6.3f' %(h1, h2))
	print(' o1,  o2  = %6.3f, %6.3f' %(o1, o2))

	E = (o1-t1)**2/2 + (o2-t2)**2/2
	print(' E  = %.7f' %E)
	if E < 0.0000001:
		break

	o1b, o2b = o1 - t1, o2 - t2
	h1b, h2b = o1b*w5+o2b*w7, o1b*w6+o2b*w8 # ➍
	w1b, w3b = i1*h1b, i1*h2b # ➐
	w2b, w4b = i2*h1b, i2*h2b # ➐
	b1b, b2b = 1*h1b, 1*h2b # ➑
	w5b, w7b = h1*o1b, h1*o2b # ➎
	w6b, w8b = h2*o1b, h2*o2b # ➎
	b3b, b4b = 1*o1b, 1*o2b # ➏
	print(' w1b, w3b = %6.3f, %6.3f'%(w1b, w3b))
	print(' w2b, w4b = %6.3f, %6.3f'%(w2b, w4b))
	print(' b1b, b2b = %6.3f, %6.3f'%(b1b, b2b))
	print(' w5b, w7b = %6.3f, %6.3f'%(w5b, w7b))
	print(' w6b, w8b = %6.3f, %6.3f'%(w6b, w8b))
	print(' b3b, b4b = %6.3f, %6.3f'%(b3b, b4b))

	lr = 0.01
	w1, w3 = w1 - lr*w1b, w3 - lr*w3b # ⓫
	w2, w4 = w2 - lr*w2b, w4 - lr*w4b # ⓫
	b1, b2 = b1 - lr*b1b, b2 - lr*b2b # ⓬
	w5, w7 = w5 - lr*w5b, w7 - lr*w7b # ➒
	w6, w8 = w6 - lr*w6b, w8 - lr*w8b # ➒
	b3, b4 = b3 - lr*b3b, b4 - lr*b4b # ➓
	print(' w1,  w3  = %6.3f, %6.3f'%(w1, w3))
	print(' w2,  w4  = %6.3f, %6.3f'%(w2, w4))
	print(' b1,  b2  = %6.3f, %6.3f'%(b1, b2))
	print(' w5,  w7  = %6.3f, %6.3f'%(w5, w7))
	print(' w6,  w8  = %6.3f, %6.3f'%(w6, w8))
	print(' b3,  b4  = %6.3f, %6.3f'%(b3, b4))