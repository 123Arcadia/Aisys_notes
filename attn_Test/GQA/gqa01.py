import torch
import math
from torch import nn, Tensor
import torch.nn.functional as F

from attn_Test.Attn.attn01 import Attention
from attn_Test.MQA.MQA_test import MultiQueryAttention

"""
代码: https://cyrilzakka.github.io/llm-playbook/nested/topk.html
"""
class  GroupedQueryAttention(Attention):
    """
    https://arxiv.org/pdf/2305.13245.pdf

    GQA-1: A single group, which equates to Multi-Query Attention (MQA).
    GQA-H: Groups equal to the number of heads, essentially the same as Multi-Head Attention (MHA).
    GQA-G: An intermediate configuration with G groups, balancing between efficiency and expressiveness.
    """
    def __init__(self, word_size: int = 512, embed_dim: int = 64,
                 n_grouped: int = 4, n_query_each_group:int=2) -> None:
        super().__init__(word_size, embed_dim)
        delattr(self, 'query')
        delattr(self, 'key')
        delattr(self, 'value')

        self.grouped = nn.ModuleList([
            MultiQueryAttention(word_size, embed_dim, n_query=n_query_each_group)
            for _ in range(n_grouped)
        ])
        # self.proj = nn.Parameter(torch.empty((..., ...), requires_grad=True))
        self.proj = nn.Parameter(torch.empty(embed_dim * n_grouped, embed_dim))
        nn.init.xavier_uniform_(self.proj)

    def forward(self, x: Tensor) -> Tensor:
        # 每个mqa使用一样的KV
        Z_s = torch.cat([mqa(x) for mqa in self.grouped], dim=1)
        Z = torch.matmul(Z_s, self.proj)
        return Z
