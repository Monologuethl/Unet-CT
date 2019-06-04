import os
import numpy as np
from cv2 import cv2


def binary_photo(path):
    src = cv2.imread(path)  # 读取图像
    print(src.dtype)
    R = src[:, :, 2]
    x, y = R.shape
    for i in range(x):
        for j in range(y):
            if R[i, j] != 0:
                R[i, j] = 255
    src_new = np.zeros(src.shape).astype("uint8")
    src_new[:, :, 0] = R
    src_new[:, :, 1] = R
    src_new[:, :, 2] = R
    cv2.imwrite(path, src_new)
    print(path)


label = r"C:\Users\Tong\Desktop\unet-CT\data\membrane\train\label"
file_list = os.listdir(label)  # 获取文件路径

for item in file_list:
    photo_path = os.path.join(os.path.abspath(label), item)
    binary_photo(photo_path)

# a1 = np.array([1,2,3,4],dtype=np.complex128)
# print(a1)
# print("数据类型",type(a1))           #打印数组数据类型
# print("数组元素数据类型：",a1.dtype) #打印数组元素数据类型
# print("数组元素总数：",a1.size)      #打印数组尺寸，即数组元素总数
# print("数组形状：",a1.shape)         #打印数组形状
# print("数组的维度数目",a1.ndim)      #打印数组的维度数目
