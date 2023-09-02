import numpy as np
import torch
import datasets
from model.SA_Unet import SA_UNet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model.unet_model import  *

# 定义数据对象
data_src = datasets.Chasedb1Datasets('CHASEDB1/')
data_loader = DataLoader(dataset=data_src, batch_size=1, num_workers=2,shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = UNet(1,1)
model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()  # 使用二分类损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_err = []
best_loss = float('inf')
torch.cuda.empty_cache()
def train(epoch):
    for idx, data in enumerate(data_loader):
        x, y = data
        x, y = x.to(device,dtype=torch.float32), y.to(device,dtype=torch.float32)
        y_pred = model(x)
        # y = torch.sigmoid(y)
        optimizer.zero_grad()
        loss = criterion(y_pred, y)
        loss.backward()
        global best_loss
        # 保存loss值最小的网络参数
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), 'best_model.pth')

        loss_err.append(best_loss)
        optimizer.step()

        # 可视化当前训练的x和y_pred
        if epoch  == 199:  # 每10个批次可视化一次
            visualize_sample(x, y_pred)

    print('[%d] loss : %.3f' % (epoch + 1, loss.item()))

def visualize_sample(x, y_pred):
    plt.figure()
    plt.subplot(121)
    plt.title('Input Image')
    plt.imshow(x[0][0].cpu().numpy(), cmap='gray')
    plt.subplot(122)
    plt.title('Model Prediction')
    plt.imshow(torch.sigmoid(y_pred[0][0]).detach().cpu().numpy(), cmap='gray')  # 使用detach分离梯度
    plt.show()

if __name__ == '__main__':

    for epoch in range(200):
        model.train()
        train(epoch)
        torch.save(model.state_dict(), 'train-01.pth')

    # 绘制损失曲线
    plt.figure()
    plt.plot(np.arange(0, 200, 1), loss_err[:200], 'b')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())

    print("Total number of parameters: %.2f" % total_params)
