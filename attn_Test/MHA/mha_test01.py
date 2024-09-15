import torch
import math
from torch import nn, Tensor
import torch.nn.functional as F

from attn_Test.Attn.attn01 import Attention


D = 10
V = 10
head_num = 8

"""
代码: https://cyrilzakka.github.io/llm-playbook/nested/topk.html
"""
class MultiheadAttention(nn.Module):
    r"""
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, word_size: int = D, embed_dim: int = V, n_head:int=head_num) -> None:
        super().__init__()
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.dim_K = torch.tensor(embed_dim)
        self.proj = nn.Parameter(torch.empty(embed_dim * n_head, embed_dim)) # hsape = [80 , 10]
        nn.init.xavier_uniform_(self.proj)
        self.multihead = nn.ModuleList([
            Attention(word_size, embed_dim) for _ in range(n_head)
        ])

    def forward(self, x: Tensor) -> Tensor:
        # 使用的K V也是一样的
        Z_s = torch.cat([attn(x) for attn in self.multihead], dim=1)
        # print(f'{Z_s=}')
        # 单个attn输出attn(x)是[1 10] , cat之后是[1 80]
        print(f'{Z_s.shape=}')
        # Z_s.shape=torch.Size([1, 80])
        Z = torch.matmul(Z_s, self.proj)
        return Z


if __name__ == '__main__':
    x = torch.randint(1, 64, (1, D), dtype=torch.float)

    model = MultiheadAttention(D, V, head_num)
    output = model(x)

    print(f'{output=}')
    print(f'{output.shape=}')

    # output=tensor([[ 37.7913, -11.2425,  41.8644,  46.2991,  10.0032,   4.2614,  -6.0698,
    #           21.0889, -44.0478,  45.5535]], grad_fn=<MmBackward0>)
    # output.shape=torch.Size([1, 10])