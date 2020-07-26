import time
import torch
from torch import nn, optim
import torchvision
from utils import *
#加载设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#定义网络
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

#实例化
net = AlexNet()
#加载数据集,并resize到224
train_iter, test_iter = load_data_fashion_mnist(batch_size=256, resize=224)
#测试函数
def vaild(data_iter):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            net.eval()
            acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
            net.train()
            n += y.shape[0]
    return acc_sum / n

def train(net, epochs, train_iter, test_iter, optimazer):
    net.to(device)
    print("train on cuda", device)
    loss = nn.CrossEntropyLoss()
    batch_count = 0
    for epoch in range(epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimazer.zero_grad()
            l.backward()
            optimazer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (net(X).argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1

        test_acc = vaild(test_iter)
        print("epoch %d, loss %4f, train_acc %3f, test_acc %3f, time %1f sec" % (epoch, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time()-start))

epochs, lr = 200, 0.01
optimazer = torch.optim.Adam(net.parameters(), lr=lr)
train(net, epochs, train_iter, test_iter, optimazer)


