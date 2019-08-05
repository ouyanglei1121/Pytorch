#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: ouyang lei time:2019/8/4
import torch
"""
拼接与拆分（merge or split）:
    cat
    stack
    split
    chunk
"""

# cat
a = torch.randn(4, 32, 8)
b = torch.randn(5, 32, 8)
c = torch.cat([a, b], dim=0)
print(c.shape)
# stack
a = torch.randn(32, 8)
b = torch.randn(32, 8)
x1 = torch.stack([a, b], dim=0)
x2 = torch.stack([a, b], dim=1)
print(x1.shape)
print(x2.shape)
"""
cat与stack的去别：
    cat只有合并的维度不一样，其他的维度都是一样的
    stack所有的维度都是一样的，在合并的维度前面加上tensor的数量
"""
# split: by len（每个单元的长度）
a = torch.randn(5, 32, 8)
aa, bb = a.split([3, 2], dim=0)
a1, a2, a3, a4, a5 = a.split(1, dim=0)  # a.split(list, dim) 如果把a中的拆分为相同的，则list是一个int，如果不同则list表示的是所有拆分的大小集合
print(aa.shape, bb.shape, a1.shape, a2.shape, a3.shape, a4.shape, a5.shape)

# chunk： by num
a = torch.randn(6, 32, 8)
b1, b2 = a.chunk(2, dim=0)
print(b1.shape, b2.shape)


