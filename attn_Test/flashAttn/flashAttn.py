import torch

# seed = 
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

"""
代码: https://www.zhihu.com/search?type=content&q=flashAttention%E5%AE%9E%E7%8E%B0
"""
NEG_INF = -1e10  # -infinity
EPSILON = 1e-10

Q_LEN = 5
K_LEN = 6
Q_BLOCK_SIZE = 3
KV_BLOCK_SIZE = 3
P_DROP = 0.2

Tr = Q_LEN // Q_BLOCK_SIZE # 把Q分为Tr = [N / Br]个Q1, ..., Qr
Tc = K_LEN // KV_BLOCK_SIZE # 把K, V分为Tc = [N / Bc]个,

Q = torch.randn(1, 1, Q_LEN, 4, requires_grad=True).to(device='cpu')
K = torch.randn(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')
V = torch.randn(1, 1, K_LEN, 4, requires_grad=True).to(device='cpu')

O = torch.zeros_like(Q, requires_grad=True)
# print('--etst01--')
# print(torch.zeros(Q.shape[:-1])) # shape:[1, 1, 5]
l = torch.zeros(Q.shape[:-1])[..., None] # shape:[1, 1, 5, 1]
m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

# step 4
Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)

# step 5 中间结果
O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

# step 6
for j in range(Tc): # Tc个小K, V
    # step 7
    Kj = K_BLOCKS[j]
    Vj = V_BLOCKS[j]
    # step 8
    for i in range(Tr): # Tr = Q_LEN // Q_BLOCK_SIZE 分解Q成block Q
        # step 9
        Qi = Q_BLOCKS[i]
        # 中间结果
        Oi = O_BLOCKS[i]
        li = l_BLOCKS[i]
        mi = m_BLOCKS[i]

        # step 10 Q*K^T
        S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi, Kj)

        # step 11 掩码矩阵相乘
        mask = S_ij.ge(0.5)
        S_ij = torch.masked_fill(S_ij, mask, value=0)
        
        # step 12
        m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True) # 最后一个维度的最大的一个元素
        P_ij = torch.exp(S_ij - m_block_ij) # 计算softmax分子
        l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON # 计算softmax分母
        P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj) #这里P_ij先和Vj相乘

        # step 13
        mi_new = torch.maximum(m_block_ij, mi) # 更新最大矩阵

        li_new = torch.exp(mi - mi_new) * li + \
                 torch.exp(m_block_ij - mi_new) * l_block_ij # 更新softmax分母

        # step 14
        m = torch.nn.Dropout(p=P_DROP)
        P_ij_Vj = m(P_ij_Vj)

        # Step 15
        O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi \
                      + (torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
        print(f'-----------Attention : Q{i}xK{j}---------')
        print(O_BLOCKS[i].shape)
        print(O_BLOCKS[0])
        print(O_BLOCKS[1])
        print('\n')

        # step 16
        l_BLOCKS[i] = li_new
        m_BLOCKS[i] = mi_new

O = torch.cat(O_BLOCKS, dim=2)
l = torch.cat(l_BLOCKS, dim=2)
m = torch.cat(m_BLOCKS, dim=2)