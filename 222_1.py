x = 2
t = 10
w = 3
b = 1

y = x*w + 1*b # ➊
print('y  = %6.3f' %y)

yb = y - t
xb = yb*w # ➋
wb = yb*x # ➌
bb = yb*1 # ➍
print('xb = %6.3f, wb = %6.3f, bb = %6.3f'%(xb, wb, bb))

lr = 0.01
w = w - lr*wb # ➎
b = b - lr*bb # ➏
print('x  = %6.3f, w  = %6.3f, b  = %6.3f'%(x, w, b))