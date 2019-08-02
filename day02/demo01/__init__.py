#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: ouyang lei time:2019/7/31

from nltk.book import *

# fdist = FreqDist(text5)
# print(sorted([w for w in set(text5) if len(w) > 7 and fdist[w] > 7]))
#
# fdist.plot(cumulative=True)

# print([w for w in sent7 if len(w) < 4])
print(sorted([w for w in set(text1) if w.endswith('ableness')]))