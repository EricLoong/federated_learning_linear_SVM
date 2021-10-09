import copy

import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random





def torch_train(x, y, args, weights=list(), bias=list(), lr_mod = 'Pegasos'):
    n_samples = x.shape[0]
    n_features = x.shape[1]

    x = torch.FloatTensor(x)
    y = torch.FloatTensor(y)

    if len(weights) == 0:
        print('Training starts from the very beginning')
        model = nn.Linear(in_features=n_features, out_features=1)

    else:
        print('Online learning mode activate, get weights and bias from historical knowledge')
        model = nn.Linear(in_features=n_features, out_features=1)
        model.weight = copy.deepcopy(weights)
        model.bias = copy.deepcopy(bias)
    if lr_mod =='Optimal':
        print('Optimal learning rate')
        t0 = 1/(args.lmd*0.01)
        lambda1 = lambda epoch: 1 / (epoch + t0)
    else:
        print('Pegasos learning rate')
        lambda1 = lambda epoch: 1 / (epoch + 2)

    learning_rate = 1 / args.lmd
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    # schedule the learning rate to Pegasos SVM learning rate
    seed_index = random.randint(0, 1234567)
    L_oss = list()
    norm_grad_list = list()
    model.train()
    for epoch in range(1, args.epoch + 1):
        torch.manual_seed(seed_index)
        permutation = torch.randperm(n_samples)
        loss_sum = 0
        epoch_grad = list()
        for i in range(0, n_samples, args.batchsize):
            x_train = x[permutation[i:i + args.batchsize]].to(torch.device("cpu"))
            y_train = y[permutation[i:i + args.batchsize]].to(torch.device("cpu"))

            optimizer.zero_grad()
            output = model(x_train.float()).squeeze()
            weight = model.weight.squeeze()
            bias = model.bias.squeeze()


            w = model.weight
            b = model.bias

            # Here take hinge loss
            loss = torch.mean(torch.clamp(1 - y_train * output, min=0))
            loss += args.lmd * (weight.t() @ weight) * 0.5

            loss.backward()
            grad_rwt_weight = model.weight.grad
            epoch_grad.append(torch.norm(grad_rwt_weight).squeeze())
            optimizer.step()

            loss_sum += float(loss)

        norm_grad_list.append(sum(epoch_grad)/(n_samples/args.batchsize))
        L = copy.deepcopy(loss_sum / n_samples)
        L_oss.append(L)
        scheduler.step()
        print('Epoch:{:5d}\tloss:{}'.format(epoch, loss_sum / n_samples))
    #print('Finish this binary SVM classfier training. And average norm grad is: ' , norm_grad_list)
    return w, b, L_oss, norm_grad_list


class SGD_SVM_Torch():

    def __init__(self, args,lr_mode='Pegasos',weights=None, bias = None):
        self.weights = weights
        self.bias = bias
        self.loss = 0
        self.args = args
        self.epoch = args.epoch
        self.batchsize = args.batchsize
        self.learning_rate = 1 / args.lmd
        self.learning_rate_mode = lr_mode
        self.weights_norm = None

    def fit(self, x, y, weights=None, bias=None, num_class=10):
        dim = x.shape[1]
        n_samples = x.shape[0]
        if type(weights) == type(None):
            W = np.zeros((num_class, dim))
            B = np.zeros((num_class, 1))
        else:
            print('Obtain parameters from previous models')
            W_copy = copy.deepcopy(weights)
            B_copy = copy.deepcopy(bias)
            W = W_copy.detach().numpy()
            B = B_copy.detach().numpy()

        self.weights = W
        self.bias = B
        self.loss = np.zeros((num_class, self.epoch))
        self.weights_norm = np.zeros((num_class, self.epoch))

        def preprocess_binary(y, target_i):
            y = np.array(y)
            y[np.where(y != target_i)] = -1
            y[np.where(y == target_i)] = 1
            return y

        for i in range(num_class):
            if i in np.unique(y):
                temp_y = preprocess_binary(y, target_i=i)
                tensor_w = nn.Parameter(torch.FloatTensor(W[i, :].reshape(1, -1)))
                tensor_b = nn.Parameter(torch.FloatTensor(B[i]))
                temp_w, temp_b, temp_loss, temp_grad_norm = torch_train(x, y=temp_y, args=self.args,
                                                        weights=tensor_w, bias=tensor_b, lr_mod=self.learning_rate_mode)
                temp_w = temp_w.detach().numpy()
                temp_b = temp_b.detach().numpy()
                W[i, :] = temp_w
                B[i, :] = temp_b
                self.loss[i, :] = temp_loss
                self.weights_norm[i,:] = temp_grad_norm
            #print('Number of classifiers finished: '+str(i+1))

                # Obtain weight and bias matrix
        self.weights = torch.FloatTensor(W)
        self.bias = torch.FloatTensor(B)

    def predict(self, x_new):
        weight = (self.weights).detach().numpy()
        bias = (self.bias).detach().numpy()
        n_s = x_new.shape[0]
        pred_c = list()
        for i in range(n_s):
            # select maximum value as the predicted class, since the decision value shows
            # the probability of class, which is also used by sklearn package
            decision_list = np.array(np.dot(x_new[i, :], weight.T)).reshape(-1, 1) + bias
            pred_class = np.where(decision_list == max(decision_list))[0][0]
            pred_c.append(pred_class)
        output = np.array(pred_c)

        return output

def get_weights(model: SGD_SVM_Torch):

    weights = model.weights
    bias = model.bias
    return weights,bias

class argp():

    def __init__(self, lmd=0.0001, epoch=100, batchsize=32):
        self.lmd = lmd
        self.epoch = epoch
        self.batchsize = batchsize