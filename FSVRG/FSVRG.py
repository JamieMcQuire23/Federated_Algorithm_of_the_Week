import syft as sy
import torch


class FSVRG:

    def __init__(self, workers, n_iter, model, step_size, loss_function):

        self.workers = workers
        self.n_iter = n_iter
        self.model = model
        self.step_size = step_size
        self.loss_function = loss_function

    def train_model(self, n_workers, federated_data_loader):

        selected_workers = self.__select_n_workers(n_workers)
        selected_id = [x.id for x in selected_workers]
        local_epochs = [0] * len(selected_id)

        self.model.train()

        model_pointers = self.__send_model_to_workers(selected_workers)

    #function that computes the estimation of the full gradient
    def __compute_full_grad(self, model_pointers, federated_data_loader):

        worker_gradients = [0] * len(self.workers) #empty list of gradients 

        for batch_idx, (data, target) in enumerate(federated_data_loader):

            self.__partial_grad(data, target, model_pointers) #add gradient to the workers sum

            for para in model.parameters()

        return sum(workers_gradients) / len(workers_gradients) #compute the full gradient 


    #function to compute the estimation of the 
    def __partial_grad(self, data, target, model_pointer):

        model = model_pointers[data.location.id] #get the workers model
        pred = model(data) #compute the output predictions
        loss = self.loss_function(pred, target) #compute the loss from the predictions and target
        loss.backward() #compute the gradient


    
