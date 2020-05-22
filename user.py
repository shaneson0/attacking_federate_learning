import functools
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import data_sets
from sklearn.model_selection import train_test_split


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def flatten_params(params):
    return np.concatenate([i.data.cpu().numpy().flatten() for i in params])


def row_into_parameters(row, parameters):
    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x,y:x*y, param.shape)
        current_data = row[offset:offset + new_size]

        param.data[:] = torch.from_numpy(current_data.reshape(param.shape))
        offset += new_size


class User:

    def __init__(self, user_id, batch_size, is_malicious, users_count, momentum, data_set=data_sets.MNIST):
        self.is_malicious = is_malicious
        self.user_id = user_id
        self.criterion = nn.NLLLoss()
        self.learning_rate = None
        self.grads = None
        self.data_set = data_set
        self.momentum = momentum
        if data_set == data_sets.MNIST:
            self.net = data_sets.MnistNet()
        elif data_set == data_sets.CIFAR10:
            self.net = data_sets.Cifar10Net()
        self.original_params = None
        dataset = self.net.dataset(True)
        # sampler = None
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=users_count, rank=user_id)
        if users_count > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=users_count, rank=user_id)

        self.train_loader = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=batch_size, shuffle=sampler is None)
        self.train_iterator = iter(cycle(self.train_loader))
        self.MetaX, self.MetaY = self.getMetaData()
        # self.MetaX = self.MetaX.tolist()
        # self.MetaY = self.MetaY.tolist()


    # Get Meta Data
    # Fix: 0.11 need be fixed
    def getMetaData(self):
        X, y = next(self.train_iterator)
        _, MetaX, _, MetaY = train_test_split(X, y, test_size=0.11, random_state=42, stratify=y)
        return MetaX, MetaY

    def train(self, data, target):
        if self.data_set == data_sets.MNIST:
            # resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
            data = data.view(-1, 28 * 28)
        else:
            b, c, h, w = data.size()
            data = data.view(b, c, h, w)
        self.optimizer.zero_grad()

        net_out = self.net(data)
        loss = self.criterion(net_out, target)
        loss.backward()
        #self.optimizer.step() # not stepping because reporting the gradients and the server is performing the step

    # user gets initial weights and learn the new gradients based on its data
    def step(self, current_params, learning_rate):
        if self.user_id == 0 and self.is_malicious:
            self.original_params = current_params.copy()
            self.learning_rate = learning_rate
        row_into_parameters(current_params, self.net.parameters())
        self.optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=self.momentum, weight_decay=5e-4)

        data, target = next(self.train_iterator)
        self.train(data, target)
        self.grads = np.concatenate([param.grad.data.cpu().numpy().flatten() for param in self.net.parameters()])



if __name__ == '__main__':
    user_id = 1
    batch_size = 83
    is_mal = False
    users_count = 10
    momentum = 0.9
    user = User(user_id, batch_size, is_mal, users_count, momentum)
    # print (user.train_loader)
    # data, target = next(self.train_iterator)
    print (len(user.train_loader))








