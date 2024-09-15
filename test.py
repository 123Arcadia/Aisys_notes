

import torch

x = torch.tensor(3.0,requires_grad=True)

y1 = x + 1
y2 = 2 * x

loss = (y1 - y2) ** 2  # 求导:-2(1-x) 当x = 3, 等于4
loss.backward()

print(loss)

# print(y1.grad)
# print(y2.grad)
# print(x.grad)
# tensor(4., grad_fn=<PowBackward0>)
# tensor(4.)


t1 = torch.tensor([1,1,-1], dtype=torch.float)
t2 = torch.arange(1, 10, dtype=torch.float).reshape(3, 3)

print(f"{t1=}")
print(f"{t2=}")

print(torch.matmul(t2, t1))
print(torch.matmul(t1, t2))
