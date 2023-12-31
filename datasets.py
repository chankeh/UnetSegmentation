import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2

from torch.utils.data import DataLoader
import random
import numpy as np



transform = transforms.Compose(
    [
        transforms.Resize ((500,480)),      # 随机裁剪图像为 224x224 像素
        # transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
        transforms.ToTensor(),           # 将图像转换为张量
        transforms.Normalize(mean=[0.45287978], std=[0.33989161 * 255])
    ]
)

# 数据处理模块
class Chasedb1Datasets(Dataset):
    def __init__(self, root: str, transforms=transform):
        super(Chasedb1Datasets, self).__init__()
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(root, "images"))]
        img_names.sort()
        self.img_list = [os.path.join(root, "images", i) for i in img_names]
        self.img_list.sort()
        manual_names = [i for i in os.listdir(os.path.join(root, '1st_label'))]
        manual_names.sort()
        self.manual = [os.path.join(root, "1st_label", i) for i in manual_names]
        self.manual.sort()

        # print(self.img_list)
        # print(self.manual)

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip


    def __getitem__(self, idx):
        # print(self.img_list[idx])
        image = cv2.imread(self.img_list[idx], cv2.IMREAD_GRAYSCALE)

        label = cv2.imread(self.manual[idx], cv2.IMREAD_GRAYSCALE)

        # 缩小图像尺寸为原来的一半
        image = cv2.resize(image, (image.shape[0] // 2, image.shape[1] // 2), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (label.shape[0] // 2, label.shape[1] // 2), interpolation=cv2.INTER_LINEAR)

        image = image[np.newaxis,:,:]
        label = label[np.newaxis,:,:]
        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        return image, label

    def __len__(self):
        return len(self.img_list)


if __name__ == '__main__':
    # 定义数据对象
    data_src = Chasedb1Datasets('CHASEDB1/')
    data_loader = DataLoader(dataset=data_src, batch_size=1, shuffle=True)

    for data in data_loader:
        x, y = data
        print(x.shape, y.shape)
