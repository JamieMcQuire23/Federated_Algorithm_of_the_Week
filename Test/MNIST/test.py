'''
Author: Jamie
Date: 20/11/2020
Description: This is a testing file for my implementations of federated learning optimization schemes
The algorithms are trained on the MNIST dataset with a simple feedforward neural network.
'''

import os
import sys
sys.path.insert(0, "C://Users//b9027741//OneDrive - Newcastle University//PhD//Programming//Federated_Algorithm_of_the_Week//")

from FedProx.fedprox import FedProx
from Test.MNIST.torch_model import FNN

import syft as sy
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

hook = sy.TorchHook(torch)
worker_1 = sy.VirtualWorker(hook, id="worker_1")
worker_2 = sy.VirtualWorker(hook, id="worker_2")
worker_3 = sy.VirtualWorker(hook, id="worker_3")
worker_4 = sy.VirtualWorker(hook, id="worker_4")
worker_5 = sy.VirtualWorker(hook, id="worker_5")
worker_6 = sy.VirtualWorker(hook, id="worker_6")
worker_7 = sy.VirtualWorker(hook, id="worker_7")
worker_8 = sy.VirtualWorker(hook, id="worker_8")
worker_9 = sy.VirtualWorker(hook, id="worker_9")

workers = [worker_1, worker_2, worker_3, worker_4, worker_5, worker_6, worker_7, worker_8, worker_9]

model = FNN()

federated_train_loader = sy.FederatedDataLoader( 
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    .federate(workers),
    batch_size=512, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=128, shuffle=True)

optimizer = torch.optim.SGD
loss = F.nll_loss

fedprox = FedProx(workers, 10, model, optimizer, 0.01, 0.5, loss)

fedprox.federated_train(10, 3, federated_train_loader, test_loader)

