import numpy as np


class ADTangent:
    """
    ADTangent 的类，这类初始化的时候有两个参数，一个是 x，表示输入具体的数值；另外一个是 dx，表示经过对自变量 x 求导后的值
    """

    def __init__(self, x, dx):
        self.x = x
        self.dx = dx

        # 重载 str 是为了方便打印的时候，看到输入的值和求导后的值


    def __str__(self):
        context = f'value:{self.x:.4f}, grad:{self.dx}'
        return context

    """操作符重载"""
    def __add__(self, other):
        if isinstance(other, ADTangent):
            x = self.x + other.x
            dx = self.dx + other.dx
        elif isinstance(other, float):
            x = self.x + other
            dx = self.dx
        else:
            return NotImplementedError
        return ADTangent(x, dx)

    def __sub__(self, other):
        if isinstance(other, ADTangent):
            x = self.x - other.x
            dx = self.dx - other.dx
        elif isinstance(other, float):
            x = self.x - other
            ex = self.dx
        else:
            return NotImplementedError
        return ADTangent(x, dx)

    def __mul__(self, other):
        if isinstance(other, ADTangent):
            x = self.x * other.x
            dx = self.x * other.dx + self.dx * other.x
        elif isinstance(other, float):
            x = self.x * other
            dx = self.dx * other
        else:
            return NotImplementedError
        return ADTangent(x, dx)

    def log(self):
        x = np.log(self.x)
        dx = 1 / self.x * self.dx
        return ADTangent(x, dx)

    def sin(self):
        x = np.sin(self.x)
        dx = self.dx * np.cos(self.x)
        return ADTangent(x, dx)


"""
 f = ADTangent.log(x) + x * y - ADTangent.sin(y) 
 
"""
x = ADTangent(x=2., dx=1)
y = ADTangent(x=5., dx=0)
f = ADTangent.log(x) + x * y - ADTangent.sin(y)
print(f)
# value:11.6521, grad:5.5