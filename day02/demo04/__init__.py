#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: ouyang lei time:2019/8/3
import torch
""" #  tensor维度的变换
    # 维度的操作有以下几个api
    view/ reshape  重新定义tensor的size
    squeeze/unsqueeze  删减/增加  维度
    transpose/t/permute  单词的交换操作/转置/多次的交换操作
    expand/repeat  维度的扩展   expand 改变理解方式，没有扩展数据   repeat实实在在的扩展了数据
"""
a = torch.rand(4, 1, 28, 28)
print(a.shape)
print(a.reshape(4, 28*28).shape)  # 数据维度的顺序非常重要
# squeeze/unsqueeze  删减/增加  维度
print(a.unsqueeze(0).shape)  # a.unsqueeze(dim)  dim表示索要插入的位置，不管负数还是正数 都是靠近0的方向插入
b = torch.randn(1, 32, 1, 1)
print(b.squeeze().shape)
print(b.squeeze(0).shape)
print(b.squeeze(-1).shape)
print(b.squeeze(1).shape)  # 只能挤压掉（删去）维度上为1的维度
f = torch.randn(4, 32, 14, 14)
print(b.expand(4, 32, 14, 14).shape)  # b.expand() 只能扩展b维度中为1的维度，不是1的维度不能扩展
print(b.expand(-1, 32, 4, -1).shape)  # 如果expand（）中的数为-1时，表示b中对应的维度不变
print(b.repeat(4, 32, 1, 1).shape)  # b.repeat(list)  中的list表示相应的维度上复制的次数
#  transpose/t/permute  单词的交换操作/转置/多次的交换操作
a = torch.rand(4, 3, 32, 32)
a1 = a.transpose(1, 3).contiguous().view(4, 3*32*32).view(4, 3, 32, 32)  # a.transpose(1, 3)表示交换a中的1和3维度
a2 = a.transpose(1, 3).contiguous().view(4, 3*32*32).view(4, 32, 32, 3)  # .contigous()固定维度的顺序
# a 的维度和a2的维度是相同的  与a1不相同
print(b.permute(0, 2, 3, 1).shape)  # 多维度交换