# shapes: (batch_size, seq_len, num_heads, head_dim)
import torch
from sympy.polys.polyconfig import query
from torch import nn, Tensor

from attn_Test.Attn.attn01 import Attention

word_size_G = 512
embed_dim_G = 512
n_query_G = 8

"""
代码: https://cyrilzakka.github.io/llm-playbook/nested/topk.html
"""
"""
每个KV都使用不同Q
"""
class MultiQueryAttention(Attention):
    r"""
    https://arxiv.org/pdf/1911.02150.pdf
    """

    def __init__(self, word_size: int = word_size_G, embed_dim: int = embed_dim_G, n_query: int = n_query_G) -> None:
        super().__init__(word_size, embed_dim)
        self.n_query = n_query
        self.proj = nn.Parameter(torch.empty(embed_dim * n_query, embed_dim))
        nn.init.xavier_normal_(self.proj)
        delattr(self, 'query')
        self.querys = nn.ModuleList([
            nn.Linear(in_features=word_size, out_features=embed_dim, bias=False) # bias=True
            for _ in range(n_query)
        ])
        self.key = nn.Linear(in_features=word_size, out_features=embed_dim, bias=False)
        self.value = nn.Linear(in_features=word_size, out_features=embed_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        K = self.key(x)
        V = self.value(x)
        Z_s = torch.cat([
            # 每个attn(公式)计算之后，cat连起来
            self.self_attention(query(x), K, V) for query in self.querys
        ], dim=1)
        Z = torch.matmul(Z_s, self.proj)
        return Z


if __name__ == '__main__':
    x = torch.randint(1, 64, (1, word_size_G), dtype=torch.float)

    model = MultiQueryAttention(word_size_G, embed_dim_G, n_query_G)
    output = model(x)

    print(f'{output=}')
    print(f'{output.shape=}')
    # output.shape=torch.Size([1, 512])
