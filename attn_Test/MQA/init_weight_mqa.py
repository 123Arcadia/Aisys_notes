

import torch
import torch.nn as nn
import torch.nn.init as init

# 创建一个 Linear 层
linear_layer = nn.Linear(10, 20)

# 初始化权重
init.xavier_normal_(linear_layer.weight)

# 查看重初始化后的权重
print("Initialized Weights:", linear_layer.weight)
print(torch.abs(torch.empty((5, 5))-torch.zeros((5, 5))))