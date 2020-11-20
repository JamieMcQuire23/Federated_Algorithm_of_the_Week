import syft as sy
from syft.frameworks.torch.fl.utils import federated_avg
import torch.optim as optim
import torch.nn.functional as F
import torch
import random

class FedAvg:


    '''
    workers: list of workers that are available in the FL system.
    n_iter: number of local iterations of SGD at the client models
    global_iter: number of global training iterations
    model: model trained during federated learning
    '''

    def __init__(self, workers, n_iter, model, learning_rate):

        self.workers = workers
        self.n_iter = n_iter
        self.model = model
        self.learning_rate = learning_rate

    #will train the federated model for one global iteration
    def train_model(self, epochs, n_workers, federated_data_loader):

        selected_workers = self.__select_n_workers(n_workers)
        selected_id = [x.id for x in selected_workers]
        local_epochs = [0] * len(selected_id)

        self.model.train()

        model_pointers = self.__send_model_to_workers(selected_workers)

        for batch_idx, (data, target) in enumerate(federated_data_loader):

            index = selected_id.index(data.location.id)
            #check that the correct worker has been selected and the local epochs is less than the predefined limit
            if ((data.location.id in selected_id) and (local_epochs[index] < epochs)):

                self.__update_local_model(data, target, model_pointers, index, local_epochs)

        self.model = self.__model_averaging(model_pointers)

    def evaluate_global_model(self, test_loader):

        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                test_loss = F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

            print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
            ))

        return (100. * correct / len(test_loader.dataset))

    def __model_averaging(self, model_pointers):

        models_local = {}

        for worker_name, model_pointer in model_pointers.items():
            models_local[worker_name] = model_pointer.get()
            model_avg = federated_avg(models_local)

        return model_avg

    def __update_local_model(self, data, target, model_pointers, index, local_epochs):

        worker = data.location
        model = model_pointers[worker.id]
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        optimizer.zero_grad()
        pred = model(data)
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()

        local_epochs[index] += 1

    def __send_model_to_workers(self, selected_workers):

        model_pointers = {}

        for worker in selected_workers:
            model_ptr = self.model.copy().send(worker)
            model_pointers[worker.id] = model_ptr

        return model_pointers



    #select a list of workers that we can use during the training process
    def __select_n_workers(self, n_workers):
        #create a list of indicies
        selected_indices = random.sample(range(len(self.workers)), n_workers)

        #return a list of selected workers
        return [self.workers[i] for i in selected_indices]

