# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 18:21:30 2016

@author: summing
"""
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import matplotlib

fontP = matplotlib.font_manager.FontProperties(
    fname='/Library/Fonts/Songti.ttc')
# fontP.set_family('SimHei')

item = [
    '希特勒回来了', '死侍', '房间', '龙虾', '大空头',
    '极盗者', '裁缝', '八恶人', '实习生', '间谍之桥',
]

user = ['五柳君', '帕格尼六', '木村静香', 'WTF', 'airyyouth',
        '橙子c', '秋月白', 'clavin_kong', 'olit', 'You_某人',
        '凛冬将至', 'Rusty', '噢！你看！', 'Aron', 'ErDong Chen']

# Rm*n m = item n = user
RATE_MATRIX = np.array(
    [[5, 5, 3, 0, 5, 5, 4, 3, 2, 1, 4, 1, 3, 4, 5],
     [5, 0, 4, 0, 4, 4, 3, 2, 1, 2, 4, 4, 3, 4, 0],
     [0, 3, 0, 5, 4, 5, 0, 4, 4, 5, 3, 0, 0, 0, 0],
     [5, 4, 3, 3, 5, 5, 0, 1, 1, 3, 4, 5, 0, 2, 4],
     [5, 4, 3, 3, 5, 5, 3, 3, 3, 4, 5, 0, 5, 2, 4],
     [5, 4, 2, 2, 0, 5, 3, 3, 3, 4, 4, 4, 5, 2, 5],
     [5, 4, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 2, 1, 0],
     [5, 4, 3, 3, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
     [5, 4, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2],
     [5, 4, 3, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
)

nmf_model = NMF(n_components=2)  # 设有2个主题
item_dis = nmf_model.fit_transform(RATE_MATRIX)
user_dis = nmf_model.components_
print(nmf_model.reconstruction_err_)

rec_mat = np.dot(item_dis, user_dis)
filter_matrix = RATE_MATRIX < 1e-8
print('重建矩阵，并过滤掉已经评分的物品：')
rec_filter_mat = (filter_matrix * rec_mat).T
print(rec_filter_mat)

rec_user = '凛冬将至'  # 需要进行推荐的用户
rec_userid = user.index(rec_user)  # 推荐用户ID
rec_list = rec_filter_mat[rec_userid, :]  # 推荐用户的电影列表

print('推荐用户的电影：')
print(np.nonzero(rec_list))


a = NMF(n_components=2)  # 设有2个主题
W = a.fit_transform(RATE_MATRIX)
H = a.components_
print(a.reconstruction_err_)

b = NMF(n_components=3)  # 设有3个主题
W = b.fit_transform(RATE_MATRIX)
H = b.components_
print(b.reconstruction_err_)

c = NMF(n_components=4)  # 设有4个主题
W = c.fit_transform(RATE_MATRIX)
H = c.components_
print(c.reconstruction_err_)

d = NMF(n_components=5)  # 设有5个主题
W = d.fit_transform(RATE_MATRIX)
H = d.components_
print(d.reconstruction_err_)