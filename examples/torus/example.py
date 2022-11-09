def torus2(x, y, z):#定义圆环2
    value = 2*y*(y**2 - 3*x**2)*(1 - z**2) + (x**2 + y**2)**2 - (9*z**2 - 1)*(1 - z**2)
    return value

def grad_torus2(x, y, z):#对torus2上的每一点求梯度，返回值为一个梯度向量
    grad_x = 4 * x * (np.power(x, 2) + np.power(y, 2)) - 12 * x * y * (1 -  np.power(z, 2))
    grad_y = 2 * (1 - np.power(z, 2)) * (np.power(y, 2) - 3 * np.power(x, 2)) + 4 * y * (np.power(x, 2) + np.power(y, 2)) + 4 * np.power(y, 2) * (1 - np.power(z, 2))
    grad_z = -4 * y * z * (np.power(y, 2) - 3 * np.power(x, 2)) - 18 * (1 - np.power(z, 2)) * z + 2 * (9 * np.power(z, 2) - 1) * z
    grad = np.array([grad_x, grad_y, grad_z])
    return grad