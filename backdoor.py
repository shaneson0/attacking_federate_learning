import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import malicious
import data_sets
import torch.backends.cudnn as cudnn
from user import flatten_params, row_into_parameters, cycle


class BackdoorAttack(malicious.Attack):
    def __init__(self, num_std, alpha, data_set, loss, backdoor, num_epochs=30, batch_size=200, learning_rate=0.1, momentum=0.9, my_print=print):
        super(BackdoorAttack, self).__init__(num_std)
        self.my_print = my_print
        self.alpha = alpha
        self.num_epochs = num_epochs

        self.loss = loss

        self.data_set = data_set
        if data_set == data_sets.MNIST:
            self.malicious_net = data_sets.MnistNet()
        elif data_set == data_sets.CIFAR10:
            self.malicious_net = data_sets.Cifar10Net()

        self.backdoor = backdoor
        self.batch_size = batch_size
        if backdoor != 'pattern':
            self.dataset = self.malicious_net.dataset(True)
            self.train_loader = torch.utils.data.DataLoader(
                self.dataset, sampler=torch.utils.data.distributed.DistributedSampler(self.dataset, num_replicas=len(self.dataset),
                                                                                 rank=backdoor-1),
                batch_size=batch_size, shuffle=False)
        else:
            self.dataset = self.malicious_net.dataset(True, BackdoorAttack.add_pattern)
            u = int(len(self.dataset) / batch_size / 10)
            self.train_loader = torch.utils.data.DataLoader(
                self.dataset, sampler=torch.utils.data.distributed.DistributedSampler(self.dataset, num_replicas=u,
                                                                                 rank=np.random.randint(u)),
                batch_size=batch_size, shuffle=False)
        self.test_loader = self.train_loader
        self.momentum = momentum


    @staticmethod
    def add_pattern(img):
        img[:, :5, :5] = 2.8
        return img

    def _attack_grads(self, grads_mean, grads_stdev, original_params, learning_rate):

        initial_params_flat = original_params - learning_rate * grads_mean # the corrected param after the user optimized, because we still want the model to improve

        mal_net_params = self.train_malicious_network(initial_params_flat)

        #Getting from the final required mal_net_params to the gradients that needs to be applied on the parameters of the previous round.
        new_params = mal_net_params + learning_rate * grads_mean
        new_grads = (initial_params_flat - new_params) / learning_rate

        new_user_grads = np.clip(new_grads, grads_mean - self.num_std * grads_stdev,
                                 grads_mean + self.num_std * grads_stdev)

        return new_user_grads

    def test_malicious_network(self, epoch, to_print=True):
        classification_loss = nn.NLLLoss()

        with torch.no_grad():
            test_loss = 0
            correct = 0
            test_len = 0.

            for data, target in self.test_loader:

                test_len += len(data)
                data, target = Variable(data), Variable(target)

                if self.backdoor == 'pattern':
                    target *= 0  # make images with the pattern always output 0
                else:
                    target = (target + 1) % 5
                if self.data_set == data_sets.MNIST:
                    data = data.view(-1, 28 * 28)

                net_out = self.malicious_net(data)

                test_loss += classification_loss(net_out, target).data.item()
                pred = net_out.data.max(1)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data).sum()

            test_loss /= test_len
            accuracy = 100. * float(correct) / test_len

            if to_print:
                self.my_print('##Test malicious net: [{}] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(epoch,
                                                                                                          test_loss,
                                                                                                          correct,
                                                                                                          test_len,
                                                                                                          accuracy))
        return accuracy

    def init_malicious_network(self, flat_params):
        # set the malicious parameters to be the same as in the main network
        row_into_parameters(flat_params, self.malicious_net.parameters())

    def train_malicious_network(self, initial_params_flat):
        self.init_malicious_network(initial_params_flat)
        initial_params = [torch.tensor(torch.empty(p.shape), requires_grad=False) for p in
                          self.malicious_net.parameters()]
        row_into_parameters(initial_params_flat, initial_params)

        initial_accuracy = self.test_malicious_network('BEFORE', to_print=False)
        if initial_accuracy >= 100.:
            return initial_params_flat

        train_len = 0
        '''Train'''
        self.malicious_net.train()

        for epoch in range(self.num_epochs):
            for data, target in self.train_loader:

                data, target = Variable(data, requires_grad=True), Variable(target)

                train_len += len(data)
                if self.backdoor == 'pattern':
                    target *= 0
                else:
                    target = (target + 1) % 5
                optimizer = optim.SGD(self.malicious_net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
                classification_loss = nn.NLLLoss()
                dist_loss_func = nn.MSELoss()

                if self.data_set == data_sets.MNIST:
                    data = data.view(-1, 28 * 28)
                optimizer.zero_grad()
                net_out = self.malicious_net(data)
                loss = classification_loss(net_out, target)
                if self.alpha > 0:
                    dist_loss = 0
                    for idx, p in enumerate(self.malicious_net.parameters()):
                        dist_loss += dist_loss_func(p, initial_params[idx])
                    if torch.isnan(dist_loss):
                        raise Exception("Got nan dist loss")

                    loss += dist_loss * self.alpha


                if torch.isnan(loss):
                    raise Exception("Got nan loss")
                loss.backward()
                optimizer.step()
            '''Test'''
            if epoch == (self.num_epochs - 1):
                self.test_malicious_network(epoch, to_print=True)

        return flatten_params(self.malicious_net.parameters())
