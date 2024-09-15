# 参考:https://zhuanlan.zhihu.com/p/667763542
import torch
import torch.nn.functional as F




class kv_cache(torch.nn.Module):
    def __init__(self, D, V):
        super().__init__()
        self.D = D
        self.V = V
        self.embedding = torch.nn.Embedding(V, D) # 64, 128
        self.Wq = torch.nn.Linear(D, D)
        self.Wk = torch.nn.Linear(D, D)
        self.Wv = torch.nn.Linear(D, D)
        self.lm_head = torch.nn.Linear(D, V)
        self.cache_k = self.caceh_v = None  # init

    def forward(self, X):
        X = self.embedding(X) # shape = D, D
        Q, K, V = self.Wq(X), self.Wk(X), self.Wv(X)
        print(f"{X.shape=}")
        print(f"{Q.shape=}")
        print(f"{K.shape=}")
        print(f"{V.shape=}")

        if self.cache_k == None:
            self.cache_k = K
            self.cache_v = V
        else:
            self.cache_k = torch.cat((self.cache_k, K), dim=1)
            self.cache_v = torch.cat((self.cache_v, V), dim=1)
            K = self.cache_k
            V = self.cache_v

        print(f"{self.cache_k.shape=}")
        print(f"{self.cache_v.shape=}")

        # ignore proj/MLP/scaled/mask/multi-head when calculate Attention
        attn = Q @ K.transpose(1,2) @ V

        output = self.lm_head(attn)

        return output


D = 128
V = 64
model = kv_cache(D, V)

X = torch.randint(0, 64, (1,10))
print(f"{X.shape=}")

for i in range(4):
    print(f"\nGeneration {i} step input_shape: {X.shape}:")
    output = model.forward(X)
    print(output.shape)
    nxt_token = torch.argmax(F.softmax(output, dim=-1), -1)[:, -1]
    print(f"{nxt_token.shape=}")
    X = nxt_token.unsqueeze(0)

# X.shape=torch.Size([1, 10])

# Generation 0 step input_shape: torch.Size([1, 10]):
# X.shape=torch.Size([1, 10, 128])
# Q.shape=torch.Size([1, 10, 128])
# K.shape=torch.Size([1, 10, 128])
# V.shape=torch.Size([1, 10, 128])
# self.cache_k.shape=torch.Size([1, 10, 128])
# self.cache_v.shape=torch.Size([1, 10, 128])
# torch.Size([1, 10, 64])
# nxt_token.shape=torch.Size([1])

# Generation 1 step input_shape: torch.Size([1, 1]):
# Q.shape=torch.Size([1, 1, 128])
# K.shape=torch.Size([1, 1, 128])
# V.shape=torch.Size([1, 1, 128])
# self.cache_k.shape=torch.Size([1, 11, 128])
# self.cache_v.shape=torch.Size([1, 11, 128])
# torch.Size([1, 1, 64])
# nxt_token.shape=torch.Size([1])

# Generation 2 step input_shape: torch.Size([1, 1]):
# Q.shape=torch.Size([1, 1, 128])
# K.shape=torch.Size([1, 1, 128])
# V.shape=torch.Size([1, 1, 128])
# self.cache_k.shape=torch.Size([1, 12, 128])
# self.cache_v.shape=torch.Size([1, 12, 128])
# torch.Size([1, 1, 64])
# nxt_token.shape=torch.Size([1])

# Generation 3 step input_shape: torch.Size([1, 1]):
# Q.shape=torch.Size([1, 1, 128])
# K.shape=torch.Size([1, 1, 128])
# V.shape=torch.Size([1, 1, 128])
# self.cache_k.shape=torch.Size([1, 13, 128])
# self.cache_v.shape=torch.Size([1, 13, 128])
# torch.Size([1, 1, 64])
# nxt_token.shape=torch.Size([1])