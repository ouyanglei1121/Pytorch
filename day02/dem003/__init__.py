#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: ouyang lei time:2019/8/3
import torch
# 索引与切片


#索引
a = torch.rand(4, 3, 28, 28)
print(a[0].shape)
print(a[0, 0].shape)
print(a[0, 2, 3, 8])
print(a[:, :1, ::2, :].shape)  # start : end : step
# 给定国定的值进行索引  a.index_select(dim, tensor)  dim为你所选择的那个维度， tensor为你要索引的的列表
print(a.index_select(0, torch.tensor([2, 3])).shape)
print(a[0, ...].shape)   # ...表示任意维度

# 通过掩码来索引  .masked_select()
x = torch.randn(3, 4)
mask = x.ge(0.5)
print(mask)
print(torch.masked_select(x, mask))  # 将x打平，选择出掩码为1的元素
# take函数也是把数据打平来处理的
src = torch.tensor([[4, 3, 5], [6, 7, 8]])
print(torch.take(src, torch.tensor([0, 2, 5])))  # torch.take(src, tensor) 首先将src打平， 然后根据给定的索引，依据打平后的顺序索引来索引元素

