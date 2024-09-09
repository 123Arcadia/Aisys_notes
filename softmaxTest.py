
import  torch

t1 = torch.arange(0, 10).to(dtype=torch.float32)
print(t1, t1.dtype, t1.device)
# tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]) torch.float32 cpu

t1_e = torch.exp(t1)
print(t1_e, t1_e.dtype, t1_e.device)

t1_e_sum = torch.sum(t1_e)
print(t1_e_sum, t1_e_sum.dtype, t1_e_sum.device)

t_softmax_ele = t1_e / t1_e_sum
print(t_softmax_ele)
print(torch.sum(t_softmax_ele))
# tensor([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]) torch.float32 cpu
# tensor([1.0000e+00, 2.7183e+00, 7.3891e+00, 2.0086e+01, 5.4598e+01, 1.4841e+02,
#         4.0343e+02, 1.0966e+03, 2.9810e+03, 8.1031e+03]) torch.float32 cpu
# tensor(12818.3086) torch.float32 cpu
# tensor([7.8013e-05, 2.1206e-04, 5.7645e-04, 1.5669e-03, 4.2594e-03, 1.1578e-02,
#         3.1473e-02, 8.5552e-02, 2.3255e-01, 6.3215e-01])
# tensor(1.0000)