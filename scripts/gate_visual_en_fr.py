# 折线图
# coding:utf-8
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

# datas
# plt.figure(figsize=(4,4))
testsets = ['Test2016', 'Test2017', 'MSCOCO']
zt = [0.9453, 0.9453, 0.9453]
rt = [0.5519, 0.5509, 0.5519]

bar_width = 0.3
index_zt = np.arange(len(testsets))
index_rt = index_zt + bar_width

# curves
plt.bar(index_zt, height=zt, width=bar_width, label='Zt')
plt.bar(index_rt, height=rt, width=bar_width,  label='Rt')

# 设置数字标签
for x, bleu in zip([index_zt, index_rt], [zt, rt]):
    for a, b in zip(x, bleu):
        plt.text(a, b, b, ha='center', va='bottom')

# 设置横坐标轴的刻度(纵坐标不需要设置)
plt.xticks(index_zt+bar_width/2,testsets)
# 横坐标的名字 # Iterations   x1000  纵坐标的名字
plt.xlabel('Test sets')
plt.ylabel('Avg. Weight')

# 图标题 #图例 , loc=”best”是自动选择放图例的合适位置
plt.legend(loc='best')

#保存图
path = r'G:\ubuntu\code\AAAI2022&ACL2022\align-mmt'
# 去除图片周围的白边
plt.savefig(path+"\gate_visual_en_fr.eps", bbox_inches='tight', dpi=1000, pad_inches=0.0)

# 显示图
plt.show()