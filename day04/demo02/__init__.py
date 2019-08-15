"""Tensor advanced operation
Where     Gather
"""
import torch
"""where"""
cond = torch.tensor([[0.122, 0.322, 0.623],
                     [0.544, 0.833, 0.343]])
a = torch.ones(2, 3)
b = torch.ones(2, 3) * 3
x = torch.where(cond > 0.5, a, b)
print(x)

print("""----------gather(收集) 查表操作的过程-----------""")
prob = torch.randn(4, 10)
print(prob)
idx = prob.topk(dim=0, k=2)
print(idx)
idx = idx[1]
print(idx)
label = torch.arange(10) + 100
print(label)
print(torch.gather(label.expand(4, 10), dim=0, index=idx.long()))




