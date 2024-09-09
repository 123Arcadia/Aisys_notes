
from typing import List, NamedTuple, Callable, Dict, Optional

import numpy as np

_name = 1
def fresh_name():
    global _name
    name = f'v{_name}'
    _name += 1
    return name

class ops:
    @staticmethod
    def ops_mul(self, other):
        # forward
        x = Variable(self.value * other.value)
        print(f'{x.name} = {self.name} * {other.name}')

        # backward
        def propagate(dl_doutputs):
            dl_dx, = dl_doutputs
            dx_dself = other  # partial derivate of r = self*other
            dx_dother = self  # partial derivate of r = self*other
            dl_dself = dl_dx * dx_dself
            dl_dother = dl_dx * dx_dother
            dl_dinputs = [dl_dself, dl_dother]
            return dl_dinputs

        # record the input and output of the op
        tape = Tape(inputs=[self.name, other.name], outputs=[x.name], propagate=propagate)
        gradient_tape.append(tape)
        return x

    @staticmethod
    def ops_add(self, other):
        x = Variable(self.value + other.value)
        print(f'{x.name} = {self.name} + {other.name}')

        def propagate(dl_doutputs):
            dl_dx, = dl_doutputs
            dx_dself = Variable(1.)
            dx_dother = Variable(1.)
            dl_dself = dl_dx * dx_dself
            dl_dother = dl_dx * dx_dother
            return [dl_dself, dl_dother]

        # record the input and output of the op
        tape = Tape(inputs=[self.name, other.name], outputs=[x.name], propagate=propagate)
        gradient_tape.append(tape)
        return x

    @staticmethod
    def ops_sub(self, other):
        x = Variable(self.value - other.value)
        print(f'{x.name} = {self.name} - {other.name}')

        def propagate(dl_doutputs):
            dl_dx, = dl_doutputs
            dx_dself = Variable(1.)
            dx_dother = Variable(-1.)
            dl_dself = dl_dx * dx_dself
            dl_dother = dl_dx * dx_dother
            return [dl_dself, dl_dother]

        # record the input and output of the op
        tape = Tape(inputs=[self.name, other.name], outputs=[x.name], propagate=propagate)
        gradient_tape.append(tape)
        return x

    @staticmethod
    def ops_sin(self):
        x = Variable(np.sin(self.value))
        print(f'{x.name} = sin({self.name})')

        def propagate(dl_doutputs):
            dl_dx, = dl_doutputs
            dx_dself = Variable(np.cos(self.value))
            dl_dself = dl_dx * dx_dself
            return [dl_dself]

        # record the input and output of the op
        tape = Tape(inputs=[self.name], outputs=[x.name], propagate=propagate)
        gradient_tape.append(tape)
        return x

    @staticmethod
    def ops_log(self):
        x = Variable(np.log(self.value))
        print(f'{x.name} = log({self.name})')

        def propagate(dl_doutputs):
            """
            各个变量方向的微分
            :param dl_doutputs: 数值
            :return: 各个微分list[]
            """
            dl_dx, = dl_doutputs
            dx_dself = Variable(1 / self.value)
            dl_dself = dl_dx * dx_dself
            return [dl_dself]

        # record the input and output of the op
        tape = Tape(inputs=[self.name], outputs=[x.name], propagate=propagate)
        gradient_tape.append(tape)
        return x

    @staticmethod
    def grad(l, results):
        dl_d = {}  # map dL/dX for all values X
        dl_d[l.name] = Variable(1.)
        print("dl_d", dl_d)
        print("results", results, [result.name for result in results]) # results [2.0, 5.0] ['v-1', 'v0']

        def gather_grad(entries):
            return [dl_d[entry] if entry in dl_d else None for entry in entries]

        print("gradient_tape:", gradient_tape)
        # gradient_tape: [
        # Tape(inputs=['v-1'], outputs=['v1'], propagate=<function ops.ops_log.<locals>.propagate at 0x7f7861a49430>),
        # Tape(inputs=['v-1', 'v0'], outputs=['v2'], propagate=<function ops.ops_mul.<locals>.propagate at 0x7f7861a494c0>),
        # Tape(inputs=['v1', 'v2'], outputs=['v3'], propagate=<function ops.ops_add.<locals>.propagate at 0x7f7861a49550>),
        # Tape(inputs=['v0'], outputs=['v4'], propagate=<function ops.ops_sin.<locals>.propagate at 0x7f7861a495e0>),
        # Tape(inputs=['v3', 'v4'], outputs=['v5'], propagate=<function ops.ops_sub.<locals>.propagate at 0x7f7861a49670>)]
        for entry in reversed(gradient_tape):
            print(entry)
            dl_doutputs = gather_grad(entry.outputs) # 变量的数值
            dl_dinputs = entry.propagate(dl_doutputs) # 各个微分
            print(f"{dl_doutputs=}")
            print(f"{dl_dinputs=}")

            for input, dl_dinput in zip(entry.inputs, dl_dinputs):
                print("input:", input, "dl_dinput:", dl_dinput)
                if input not in dl_d: # 如果该输入不是输出变量
                    dl_d[input] = dl_dinput # dl_d加入该变量的微分结果
                else:
                    # 进行梯度累积，反向传播给上一次的操作计算
                    dl_d[input] += dl_dinput # 该输出变量 += 输入的微分 类似: y + dx
        print(dl_d) # {'v5': 1.0, 'v3': 1.0, 'v4': -1.0, 'v0': 1.7163378145367738, 'v1': 1.0, 'v2': 1.0, 'v-1': 5.5}
        for name, value in dl_d.items():
            print(f'd{l.name}_d{name} = {value.name} = {value.value}')

        print("gather_grad:", gather_grad(result.name for result in results)) # gather_grad: [5.5, 1.7163378145367738]
        return gather_grad(result.name for result in results)




class Variable:
    def __init__(self, value, name=None):
        self.value = value
        self.name = name or fresh_name()

    def __repr__(self):
        return repr(self.value)

    # We need to start with some tensors whose values were not computed
    # inside the autograd. This function constructs leaf nodes.
    @staticmethod
    def constant(value, name=None):
        var = Variable(value, name)
        print(f'{var.name} = {value}')
        return var

    # Multiplication of a Variable, tracking gradients
    def __mul__(self, other):
        return ops.ops_mul(self, other)

    def __add__(self, other):
        return ops.ops_add(self, other)

    def __sub__(self, other):
        return ops.ops_sub(self, other)

    def sin(self):
        return ops.ops_sin(self)

    def log(self):
        return ops.ops_log(self)


###########################################



class Tape(NamedTuple):
    """
    接下来需要跟踪 Variable 所有计算，以便向后应用链式规则。那么数据结构 Tape 有助于实现这一点。
    反向传播使用链式规则，将函数的输出梯度传播给输入。其输入为 dL/dOutputs，输出为 dL/dinput。Tape 只是一个记录所有计算的累积 List 列表
    """
    inputs : List[str]
    outputs : List[str]
    # apply chain rule
    propagate : "Callable[List[Variable], List[Variable]]"


"""
记录每次 Variable 执行计算的顺序，Tape 这里面主要是记录正向的计算，把输入、输出和执行运算的操作符记录下来
"""
gradient_tape: List[Tape] = []

# reset tape
def reset_tape():
    """
    重置 Tape 的方法 reset_tape，方便运行多次自动微分，每次自动微分过程都会产生 Tape List
    """
    global _name
    _name = 1
    gradient_tape.clear()



if __name__ == '__main__':
    reset_tape()

    x = Variable.constant(2., name='v-1')
    y = Variable.constant(5., name='v0')

    f = Variable.log(x) + x * y - Variable.sin(y)
    print(f'{f=}')
    # v-1 = 2.0
    # v0 = 5.0
    # v1 = log(v-1)
    # v2 = v-1 * v0
    # v3 = v1 + v2
    # v4 = sin(v0)
    # v5 = v3 - v4
    # f=11.652071455223084

    print('#############反向################')
    dx, dy = ops.grad(f, [x, y])

    print("dx", dx)
    print("dy", dy)

