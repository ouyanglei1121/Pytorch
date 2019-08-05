#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: ouyang lei time:2019/8/4
import torch
"""
统计属性：
        norm   范数
        mean sum  均值  和
        prod
        max, min, argmin, argmax  最大值， 最小值， 最小值的位置， 最大值的位置
        kthvalue, topk
"""
# norm()
print('------范数---------')
a = torch.full([8], 1)*3
b = a.view(2, 4)
c = a.view(2, 2, 2)
print(a)
print(b)
print(c)
print(b.norm(1))
print(b.norm(2, dim=1))  # norm(int, dim)  int为范数， dim为维度， 去哪个维度的范数，哪个维度就会被消掉
print(c.norm(2, dim=0))
print(b.norm(2, dim=1).shape)
print(c.norm(2, dim=0).shape)
# mean, sum, min, max, prod(累乘)
a = torch.arange(1, 9).view(2, 4).float()
print(a)
print(a.min())
print('a.max', a.max(dim=0))
print(a.mean())
print(a.prod())
print(a.argmin())  # 不指定维度的话，首先将tensor打平，返回的是打平后的索引值
print(a.argmax())
a = torch.randn(4, 10)
print(a.argmax(dim=1))  # 确定维度，返回的是这个维度上的最大大值的索引
print(a.argmax(dim=0))  # 确定维度，返回的是这个维度上的最大大值的索引


