import time
import torch
from torch import nn, optim
import torchvision
from utils import *
#加载设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#定义卷积块
def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(2, 2)) # 这里会使宽高减半
    return nn.Sequential(*blk)

#定义网络
ratio = 8
conv_arch = ((1, 1, 64//ratio), (1, 64//ratio, 128//ratio), (2, 128//ratio, 256//ratio), (2, 256//ratio, 512//ratio), (2, 512//ratio, 512//ratio))
fc_features = 512 * 7 * 7
fc_hidden_units = 4096
def vgg(conv_arch, fc_features, fc_hidden_units=4096):
    net = nn.Sequential()
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module("vgg_block_" + str(i+1), vgg_block(num_convs, in_channels, out_channels))
    net.add_module("fc", nn.Sequential(FlattenLayer(),
                                       nn.Linear(fc_features, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, 10)))
    return net
#实例化
net = vgg(conv_arch, fc_features//ratio, fc_hidden_units//ratio)

#加载数据集,并resize到224
train_iter, test_iter = load_data_fashion_mnist(batch_size=64, resize=224)
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

epochs, lr = 200, 0.001
optimazer = torch.optim.Adam(net.parameters(), lr=lr)
train(net, epochs, train_iter, test_iter, optimazer)

