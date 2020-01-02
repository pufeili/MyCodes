import torch

'''
对于中间变量z，hook 的使用方式为：z.register_hook(hook_fn)，
其中 hook_fn为一个用户自定义的函数，其签名为：
hook_fn(grad) -> Tensor or None
'''

x = torch.Tensor([0, 1, 2, 3]).requires_grad_()
y = torch.Tensor([4, 5, 6, 7]).requires_grad_()
w = torch.Tensor([1, 2, 3, 4]).requires_grad_()
z = x + y

# flag=True
flag=False
'''
只有叶子节点的变量保存了梯度，中间变量的梯度在反向传播后自动释放
使用z.retain_grad()保存
'''
if flag:
    # z.retain_grad()

    o = w.matmul(z)
    o.backward()
    # o.retain_grad()

    print('x.requires_grad:', x.requires_grad) # True
    print('y.requires_grad:', y.requires_grad) # True
    print('z.requires_grad:', z.requires_grad) # True
    print('w.requires_grad:', w.requires_grad) # True
    print('o.requires_grad:', o.requires_grad) # True
    print('==========================================')
    print('x.grad:', x.grad) # tensor([1., 2., 3., 4.])
    print('y.grad:', y.grad) # tensor([1., 2., 3., 4.])
    print('w.grad:', w.grad) # tensor([ 4., 6., 8., 10.])
    print('z.grad:', z.grad) # None
    print('o.grad:', o.grad) # None

# flag=True
flag=False
'''
上面的方案会占用内存,使用hook函数显示中间的变量
下面这段代码hook_fn不会改变梯度值，仅仅用于显示print
'''
if flag:
    z = x + y
    # ===================
    def hook_fn(grad):
        print(grad)

    z.register_hook(hook_fn)
    # ===================
    o = w.matmul(z)
    print('=====Start backprop=====')
    o.backward()
    print('=====End backprop=====')

    print('x.grad:', x.grad)
    print('y.grad:', y.grad)
    print('w.grad:', w.grad)
    print('z.grad:', z.grad)

# flag=True
flag=False
'''
z_grad = tensor([1., 2., 3., 4.])
使用hook函数改变中间的变量,同时x,y也会受影响
'''
if flag:
    # ===================
    def hook_fn(grad):
        g = 2 * grad
        print(g)
        return g
    z.register_hook(hook_fn)
    # ===================
    o = w.matmul(z)
    print('=====Start backprop=====')
    o.backward()
    print('=====End backprop=====')
    print('x.grad:', x.grad)
    print('y.grad:', y.grad)
    print('w.grad:', w.grad)
    print('z.grad:', z.grad)






