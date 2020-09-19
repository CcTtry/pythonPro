# @Project -> File   ：flyTest -> draw
# @IDE    ：PyCharm
# @Author ：Ctry
# @Date   ：2020/9/15 21:31
# @Desc   ：

node = [0, 503, 294, 91, 607, 540, 250, 340, 277, 612]

import matplotlib.pyplot as plt
import  pandas as pd
from  mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot, savefig

excel1 = pd.read_excel("数据集1终稿.xlsx")
coordinates_2 = excel1.values[1:, 1:4]  # 全部行 2-5列

correction_vector = list(excel1.values[1:, 4])  # 取所有行的第5列
correction_vector[0] = 2
correction_vector[612] = 3
fig = plt.figure()
ax = Axes3D(fig)

for i in range(613):
    if i == 0:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='y')
    elif 0 < i < 612 and correction_vector[i] == 1:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='g', marker='+')
    elif 0 < i < 612 and correction_vector[i] == 0:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='b', marker='^')
    else:
        ax.scatter(coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2], c='y')
        ax.scatter(list(coordinates_2[1:613, 0]), list(coordinates_2[1:613, 1]),
                   list(coordinates_2[1:613, 2]), c='y', marker='.')
    print("i={} x={} y={} z={} ".format(i, coordinates_2[i, 0], coordinates_2[i, 1], coordinates_2[i, 2]))
ax.scatter(0, 50000, 5000, c='r')
ax.scatter(100000, 59652.3433795158, 5022.00116448164, c='g')
for i in range(len(node) - 1):
    ax.plot([coordinates_2[node[i], 0], coordinates_2[node[i + 1], 0]],
            [coordinates_2[node[i], 1], coordinates_2[node[i + 1], 1]],
            [coordinates_2[node[i], 2], coordinates_2[node[i + 1], 2]],
            c='k', linewidth=1)
savefig("./test.jpg")
plt.show()


