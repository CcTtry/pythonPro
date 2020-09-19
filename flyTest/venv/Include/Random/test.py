# @Project -> File   ：flyTest -> test
# @IDE    ：PyCharm
# @Author ：Ctry
# @Date   ：2020/9/17 10:46
# @Desc   ：
import numpy as np


import numpy as np
arrayA = np.array([[1, 2], [3, 4]])
arrayB = np.array([[1, 2],  [3, 4]])
arrayC = arrayB*2
print('A*B \n', arrayA*arrayB)
print('np.dot(A,B)\n', np.dot(arrayA, arrayB))
print('对应位置相乘 \n', np.multiply(arrayA, arrayB))
print('arrayc \n', arrayC)
print('对应位置相乘 \n', np.multiply(arrayA, arrayC))

