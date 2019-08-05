#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: ouyang lei time:2019/8/4

import torch
"""
Broadcast 自动扩展
    1.expand
    2.without copying data(不需要考虑数据)
"""
a = torch.randn(4, 3, 32, 32)
print(a.shape)
b = torch.randn(3, 1, 1)
print(b.shape)
c = a + b
print(c.shape)
