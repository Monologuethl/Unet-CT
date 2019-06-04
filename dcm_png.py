import os
import numpy as np
from cv2 import cv2


def dcm_png(path):
    src = cv2.imread(path)  # 读取图像
    B = src[:, :, 0]
    G = src[:, :, 1]
    R = src[:, :, 2]

    src_new = np.zeros(src.shape).astype("uint8")
    src_new[:, :, 0] = R
    src_new[:, :, 1] = R
    src_new[:, :, 2] = R
    cv2.imwrite(path, src_new)
    print(path)


label = r"C:\Users\Tong\Desktop\unet-CT\data\membrane\test"
file_list = os.listdir(label)  # 获取文件路径

for item in file_list:
    photo_path = os.path.join(os.path.abspath(label), item)
    dcm_png(photo_path)
cv2.waitKey(0)
cv2.destroyAllWindows()
