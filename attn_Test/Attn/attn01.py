import torch
import math
from torch import nn, Tensor
import torch.nn.functional as F

# QKV的维度(D, V)
D = 10
V = 10

"""
代码: https://cyrilzakka.github.io/llm-playbook/nested/topk.html
"""
class Attention(nn.Module):
    def __init__(self, word_size: int = D, embed_dim: int = V) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.dim_K = torch.tensor(embed_dim)
        self.query = nn.Linear(in_features=word_size, out_features=embed_dim, bias=False)
        self.key = nn.Linear(in_features=word_size, out_features=embed_dim, bias=False)
        self.value = nn.Linear(in_features=word_size, out_features=embed_dim, bias=False)

    def self_attention(self, Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
        K_T = torch.transpose(K, 0, 1)
        score = torch.matmul(Q, K_T) / torch.sqrt(self.dim_K)
        score = torch.softmax(score, dim=-1)
        Z = torch.matmul(score, V)
        return Z

    def forward(self, x: Tensor) -> Tensor:
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        Z = self.self_attention(Q, K, V)
        return Z


if __name__ == '__main__':
    x = torch.randint(1, 64, (1, D), dtype=torch.float)

    model = Attention(D, V)
    output = model(x)

    print(f'{output=}')
    print(f'{output.shape=}')
# output=tensor([[-49.6490, -32.6230, -11.9152,  -2.3535, -23.5391,  30.6482,  28.8495,
#            6.4249, -37.7045, -15.2670]], grad_fn=<MmBackward0>)
# output.shape=torch.Size([1, 10])



