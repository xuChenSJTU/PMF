from __future__ import print_function
import numpy as np
from numpy.random import RandomState
import pickle
import os
import copy
from evaluations import *
class PMF():
    '''
    a class for this Double Co-occurence Factorization model
    '''
    # initialize some paprameters
    def __init__(self, R, lambda_alpha=1e-2, lambda_beta=1e-2, latent_size=50, momuntum=0.8,
                 lr=0.001, iters=200, seed=None):
        self.lambda_alpha = lambda_alpha
        self.lambda_beta = lambda_beta
        self.momuntum = momuntum
        self.R = R
        self.random_state = RandomState(seed)
        self.iterations = iters
        self.lr = lr
        self.I = copy.deepcopy(self.R)
        self.I[self.I != 0] = 1

        self.U = 0.1*self.random_state.rand(np.size(R, 0), latent_size)
        self.V = 0.1*self.random_state.rand(np.size(R, 1), latent_size)


    def loss(self):
        # the loss function of the model
        loss = np.sum(self.I*(self.R-np.dot(self.U, self.V.T))**2) + self.lambda_alpha*np.sum(np.square(self.U)) + self.lambda_beta*np.sum(np.square(self.V))
        return loss
    def predict(self, data):
        index_data = np.array([[int(ele[0]), int(ele[1])] for ele in data], dtype=int)
        u_features = self.U.take(index_data.take(0, axis=1), axis=0)
        v_features = self.V.take(index_data.take(1, axis=1), axis=0)
        preds_value_array = np.sum(u_features*v_features, 1)
        return preds_value_array

    def train(self, train_data=None, vali_data=None):
        '''
        # training process
        :param train_data: train data with [[i,j],...] and this indacates that K[i,j]=rating
        :param lr: learning rate
        :param iterations: number of iterations
        :return: learned V, T and loss_list during iterations
        '''
        train_loss_list = []
        vali_rmse_list = []
        last_vali_rmse = None

        # monemtum
        momuntum_u = np.zeros(self.U.shape)
        momuntum_v = np.zeros(self.V.shape)

        for it in range(self.iterations):
            # derivate of Vi
            grads_u = np.dot(self.I*(self.R-np.dot(self.U, self.V.T)), -self.V) + self.lambda_alpha*self.U

            # derivate of Tj
            grads_v = np.dot((self.I*(self.R-np.dot(self.U, self.V.T))).T, -self.U) + self.lambda_beta*self.V

            # update the parameters
            momuntum_u = (self.momuntum * momuntum_u) + self.lr * grads_u
            momuntum_v = (self.momuntum * momuntum_v) + self.lr * grads_v
            self.U = self.U - momuntum_u
            self.V = self.V - momuntum_v

            # training evaluation
            train_loss = self.loss()
            train_loss_list.append(train_loss)

            vali_preds = self.predict(vali_data)
            vali_rmse = RMSE(vali_data[:,2], vali_preds)
            vali_rmse_list.append(vali_rmse)

            print('traning iteration:{: d} ,loss:{: f}, vali_rmse:{: f}'.format(it, train_loss, vali_rmse))

            if last_vali_rmse and (last_vali_rmse - vali_rmse) <= 0:
                print('convergence at iterations:{: d}'.format(it))
                break
            else:
                last_vali_rmse = vali_rmse

        return self.U, self.V, train_loss_list, vali_rmse_list
