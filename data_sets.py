import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision import datasets, transforms
import math


MNIST = 'MNIST'
CIFAR10 = 'CIFAR10'
CIFAR100 = 'CIFAR100'


class MnistNet(nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def dataset(self, is_train, transform=None):
        t = [transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
        if transform:
            t.append(transform)
        return datasets.MNIST('./mnist_data', download=True, train=is_train, transform=transforms.Compose(t))


class Cifar10Net(nn.Module):
    def __init__(self):
        super(Cifar10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(16, 64, 4)
        self.pool2 = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(64 * 1 * 1, 384)
        self.fc2 = nn.Linear(384, 192)
        self.fc3 = nn.Linear(192, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

    @staticmethod
    def dataset(is_train, transform=None):
        t = [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        if transform:
            t.append(transform)
        return datasets.CIFAR10(root='./cifar10_data', download=True, train=is_train,
                                       transform=transforms.Compose(t))


#for resnet
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class Cifar100Net(nn.Module):
    def __init__(self, depth=40, num_classes=100, widen_factor=4, dropRate=0.0):
        super(Cifar100Net, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.relu(self.bn1(x))
        x = F.avg_pool2d(x, 8)
        x = x.view(-1, self.nChannels)

        return F.log_softmax(self.fc(x), dim=1)


    @staticmethod
    def dataset(is_train, transform=None):
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

        if is_train:
            t = [transforms.ToTensor(),
                    transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                      (4, 4, 4, 4), mode='reflect').squeeze()),
                    transforms.ToPILImage(),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
        else:
            t = [transforms.ToTensor(),
                 normalize]
        if transform:
            t.append(transform)

        return datasets.CIFAR100(root='./cifar100_data', train=is_train, download=True, transform=transforms.Compose(t))

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

# sklearn.model_selection.train_test_split
import numpy as np


if __name__ == '__main__':
    net = MnistNet()
    dataset = net.dataset(True)
    # sampler = None
    # sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=users_count, rank=user_id)
    users_count = 10
    batch_size = 83
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=10, rank=1)

    train_loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size, shuffle=sampler is None)
    train_iterator = iter(cycle(train_loader))
    X, y = next(train_iterator)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.11, random_state = 42, stratify = y)


    print (X_test, y_test)
    print (len(X), len(y))
    print (len(train_loader))














