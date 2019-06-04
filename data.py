from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans

Sky = [128, 128, 128]
Building = [128, 0, 0]
Pole = [192, 192, 128]
Road = [128, 64, 128]
Pavement = [60, 40, 222]
Tree = [128, 128, 0]
SignSymbol = [192, 128, 128]
Fence = [64, 64, 128]
Car = [64, 0, 128]
Pedestrian = [64, 64, 0]
Bicyclist = [0, 128, 192]
Unlabelled = [0, 0, 0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                       Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])


def adjustData(img, mask, flag_multi_class, num_class):
    if (flag_multi_class):  # 此程序中不是多类情况，所以不考虑这个
        img = img / 255
        mask = mask[:, :, :, 0] if (len(mask.shape) == 4) else mask[:, :, 0]
        # if else的简洁写法，一行表达式，为真时放在前面，不明白mask.shape=4的情况是什么，由于有batch_size，所以mask就有3维[batch_size,wigth,heigh],估计mask[:,
        # :,0]是写错了，应该写成[0,:,:],这样可以得到一片图片，
        new_mask = np.zeros(mask.shape + (num_class,))
        # np.zeros里面是shape元组，此目的是将数据厚度扩展到num_class层，以在层的方向实现one-hot结构

        for i in range(num_class):
            # for one pixel in the image, find the class in mask and convert it into one-hot vector index = np.where(
            # mask == i) index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (
            # len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i) new_mask[
            # index_mask] = 1
            new_mask[mask == i, i] = 1  # 将平面的mask的每类，都单独变成一层，
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2],
                                         new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (
        new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask
    elif (np.max(img) > 1):
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask


# 上面这个函数主要是对训练集的数据和标签的像素值进行归一化


def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="rgb", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(  # https://blog.csdn.net/nima1994/article/details/80626239
        train_path,  # 训练数据文件夹路径
        classes=[image_folder],  # 类别文件夹,对哪一个类进行增强
        class_mode=None,  # 不返回标签
        color_mode=image_color_mode,  # 灰度，单通道模式
        target_size=target_size,  # 转换后的目标图片大小
        batch_size=batch_size,  # 每次产生的（进行转换的）图片张数
        save_to_dir=save_to_dir,  # 保存的图片路径
        save_prefix=image_save_prefix,  # 生成图片的前缀，仅当提供save_to_dir时有效
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)  # 组合成一个生成器
    for (img, mask) in train_generator:
        # 由于batch是2，所以一次返回两张，即img是一个2张灰度图片的数组，[2,256,256]
        img, mask = adjustData(img, mask, flag_multi_class, num_class)  # 返回的img依旧是[2,256,256]
        yield (img, mask)


# 每次分别产出两张图片和标签，不懂yield的请看https://blog.csdn.net/mieleizhi0522/article/details/82142856

# 上面这个函数主要是产生一个数据增强的图片生成器，方便后面使用这个生成器不断生成图片


def testGenerator(test_path, num_image=30, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%d.png" % i), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img, target_size)
        # img = np.reshape(img, img.shape + (1,)) if (not flag_multi_class) else img
        img = np.reshape(img, (1,) + img.shape)
        # 将测试图片扩展一个维度，与训练时的输入[2,256,256]保持一致
        yield img


# 上面这个函数主要是对测试图片进行规范，使其尺寸和维度上和训练图片保持一致

def geneTrainNpy(image_path, mask_path, flag_multi_class=False, num_class=2, image_prefix="image", mask_prefix="mask",
                 image_as_gray=True, mask_as_gray=True):
    image_name_arr = glob.glob(os.path.join(image_path, "%s*.png" % image_prefix))
    # 相当于文件搜索，搜索某路径下与字符匹配的文件https://blog.csdn.net/u010472607/article/details/76857493/
    image_arr = []
    mask_arr = []
    for index, item in enumerate(image_name_arr):  # enumerate是枚举，输出[(0,item0),(1,item1),(2,item2)]
        img = io.imread(item, as_gray=image_as_gray)
        img = np.reshape(img, img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path, mask_path).replace(image_prefix, mask_prefix), as_gray=mask_as_gray)
        # 重新在mask_path文件夹下搜索带有mask字符的图片（标签图片）
        mask = np.reshape(mask, mask.shape + (1,)) if mask_as_gray else mask
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)  # 转换成array
    return image_arr, mask_arr


# 该函数主要是分别在训练集文件夹下和标签文件夹下搜索图片，然后扩展一个维度后以array的形式返回，是为了在没用数据增强时的读取文件夹内自带的数据


def labelVisualize(num_class, color_dict, img):
    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    # 变成RGB空间，因为其他颜色只能再RGB空间才会显示
    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]
    # 为不同类别涂上不同的颜色，color_dict[i]是与类别数有关的颜色，img_out[img == i,:]是img_out在img中等于i类的位置上的点
    return img_out / 255


# 上面函数是给出测试后的输出之后，为输出涂上不同的颜色，多类情况下才起作用，两类的话无用

def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):
    for i, item in enumerate(npyfile):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        # 多类的话就图成彩色，非多类（两类）的话就是黑白色
        io.imsave(os.path.join(save_path, "%d_predict.png" % i), img)