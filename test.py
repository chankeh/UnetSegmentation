import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model.SA_Unet import SA_UNet  # 根据你的模型导入正确的模块
from datasets import Chasedb1Datasets  # 根据你的数据集导入正确的模块

# 加载已经训练好的模型权重
model = SA_UNet()  # 实例化与训练时相同的模型结构
model.load_state_dict(torch.load('train-01.pth'))  # 加载模型权重
model.eval()  # 设置模型为评估模式

# 准备测试数据集
data_src = Chasedb1Datasets('CHASEDB1/')
data_loader = DataLoader(dataset=data_src, batch_size=1, shuffle=True)

# 定义阈值（可以根据需要调整）
threshold = 0.5

# 循环遍历数据加载器，对测试数据进行预测并可视化
for i, data in enumerate(data_loader):
    x, y = data
    # 使用模型进行预测
    with torch.no_grad():
        output = model(x)

    # 将特征图转换为二进制分割掩码
    binary_mask = (output > threshold).float()

    # 创建一个新的图形
    plt.figure(figsize=(12, 4))

    # 可视化输入图像
    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(x[0].squeeze().cpu().numpy(), cmap='gray')

    # 可视化模型的预测结果
    plt.subplot(1, 3, 2)
    plt.title('Model Prediction (Feature Map)')
    plt.imshow(output[0].squeeze().cpu().numpy(), cmap='gray')

    # 可视化二进制分割掩码
    plt.subplot(1, 3, 3)
    plt.title('Binary Segmentation Mask')
    plt.imshow(binary_mask[0].squeeze().cpu().numpy(), cmap='gray')

    # 显示当前图形
    plt.show()
