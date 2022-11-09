def plane(x, y, z):#定义一个平面（疑问：为什么要这样定义？）
    value = x - 1./3.
    return value

def grad_plane(x, y, z):#求平面的梯度
    return np.array([1., 0., 0.])