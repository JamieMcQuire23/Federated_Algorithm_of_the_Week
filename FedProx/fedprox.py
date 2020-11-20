'''
Author: Jamie McQuire
Date: 20/11/2020
Description: Syft implementation of the federated proximal (FedProx) algorithm
'''


import syft as sy
import torch
import torch.nn.functional as F
import random

class FedProx:

    '''
    FedProx optimization strategy 
    '''

    def __init__(self, workers, n_iter, model, optimizer, learning_rate, mu, loss):
        '''
        workers: list of workers in the system.
        n_iter: number of iterations of local optimization.
        model: neural network.
        optimizer: selected optimizer
        learning_rate: step wise gradient descent parameter.
        mu: proximal term hyperparameter.
        loss: loss function
        '''
        self.workers = workers
        self.n_iter = n_iter
        self.model = model
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.mu = mu 
        self.loss = loss

    def federated_train(self, epochs, n_workers, federated_data_loader, test_loader):
        '''
        description: function to use FedProx to train the federated model.
        epochs: number of epochs of federated training.
        n_workers: number of workers selected at the start of training.
        federated_data_loader: syft federated data loader
        '''
        print("Performing FedProx for {:d} iterations".format(epochs))
        self.__server_test(test_loader)
        #for each training iteration
        for epoch in range(epochs-1):

            print("FedProx iteration: {:d}".format(epoch))

            #select the workers to take part in training 
            selected_workers = self.__select_n_workers(n_workers)

            #create an empty list of local epochs for each model
            local_epochs = [0] * len(selected_workers)

            #create a list for the IDs of the selected workers
            selected_workers_id = [x.id for x in selected_workers]

            #set the model weights for training
            self.model.train()

            #send the initial model to the workers for reference in the proximal term
            init_model_pointers = self.__send_model_to_workers(selected_workers)

            #send the model to the workers 
            model_pointers = self.__send_model_to_workers(selected_workers)

            #iterating through the federated data loader but only computing based 
            for batch_idx, (data, targets) in enumerate(federated_data_loader):

                #find the index in the list of the data to record the local epochs
                if (data.location.id in selected_workers_id):
                    index = selected_workers_id.index(data.location.id)

                    #conditonal to check the data belongs to a worker we require and the local epochs is not exceeded
                    if (local_epochs[index] < self.n_iter):

                        #perform local model training with the proximal loss function
                        self.__update_local_model(data, targets, init_model_pointers, model_pointers, index, local_epochs)

            #update the global model at the end of FedProx with aggregated weights
            self.__aggregate_model_weights(model_pointers)

            self.__server_test(test_loader)


    def __server_test(self, test_loader):
        '''
        description: evaluate the federated model at the central server
        test_loader: centralized testing set
        '''

        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = self.model(data)
                test_loss = self.loss(output, target, reduction='sum').item()
                pred = output.argmax(1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)

            print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)
            ))

    
    def __aggregate_model_weights(self, model_pointers):
        '''
        description: aggregates the model weights following each iteration of FedProx
        model_pointers: dictionary of model pointers for the selected workers
        '''
        #empty list containing the local models

        model_weights = []

        #iterate through the model pointers dict
        for _, model_pointer in model_pointers.items():

            #return the local model from the worker
            local_model = model_pointer.get()

            #get a list containing the local model weghts
            local_weights = [x for x in local_model.parameters()]

            #add to the list containing a list of a models weights
            model_weights.append(local_weights)
            
        summed_weights = []

        #iterate through the different model weights
        for i in range(len(model_weights[0])):

            #set the initial weight tensor from the first model
            agg_weight = model_weights[0][i]

            #iterate through the different  odels
            for j in range(len(model_weights)):

                #sum the weights for the different models 
                agg_weight = torch.add(agg_weight, model_weights[j][i])

            #add the summed weights to model
            agg_weight = (1/len(model_weights)) * agg_weight
            summed_weights.append(agg_weight)


        #update the model weights according to the sum 
        with torch.no_grad():
            for i, params in enumerate(self.model.parameters()):
                params.copy_(summed_weights[i])

        
    def __update_local_model(self, data, targets, init_model_pointers, model_pointers, index, local_epochs):
        '''
        description: performs one round of local batch gradient descent.
        data: batch data.
        targets: associated labels.
        init_model: pointers to the initial model on the workers at start of FedProx
        model_pointers: dictionary containing. 
        index: index of the worker in the seleced workers list.
        local_epochs: number of local iterations 
        '''
        worker = data.location
        local_model = model_pointers[worker.id]
        init_model = init_model_pointers[worker.id]
        optimizer = self.optimizer(local_model.parameters(), self.learning_rate)
        optimizer.zero_grad()
        preds = local_model(data)
        loss = self.__fedprox_loss(init_model, local_model, preds, targets)
        loss.backward()
        optimizer.step()

        #append the number of local iterations for the selected worker
        local_epochs[index] += 1


    def __fedprox_loss(self, init_model, local_model, preds, targets):
        '''
        description: custom loss function for FedProx
        init_model: initial (global) model
        model: updated model
        preds: model predictions
        targets: data labels
        '''
        #compute the normal loss
        f_loss = self.loss(preds, targets)

        #create a list with the inital/updated model weights
        init_weights = [x for x in init_model.parameters()]
        updated_weights = [x for x in local_model.parameters()]

        #calculate the prox loss for all of the weights
        prox_loss = 0

        #iterate through the different weights
        for i in range(len(init_weights)):

            #add the calculated loss for each weight to the total proximal loss
            prox_loss += (self.mu / 2) * (torch.sum(torch.pow((updated_weights[i] - init_weights[i]),2)))

        return f_loss + prox_loss

    def __select_n_workers(self, n_workers):
        '''
        descripton: selects the subset of workers to take part in training
        n_workers: number of workers selected for training
        '''
        #create a list of indicies
        selected_indices = random.sample(range(len(self.workers)), n_workers)

        #return a list of selected workers
        return [self.workers[i] for i in selected_indices]


    def __send_model_to_workers(self, selected_workers):
        '''
        description: sends the model weights to the selected workers and returns a dict of 
        the pointers to the local models
        selected_workers: workers selected to take part in training
        '''

        model_pointers = {}

        for worker in selected_workers:
            model_ptr = self.model.copy().send(worker)
            model_pointers[worker.id] = model_ptr

        return model_pointers

