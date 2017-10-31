from __future__ import print_function
import torch.utils.data
import matplotlib.pyplot as plt
from evaluations import *
from pmf_model import *
from my_utils import *
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import svm

print('PMF Recommendation Model Example')

# choose dataset name and load dataset
dataset = 'ml-100k'
processed_data_path = os.path.join(os.getcwd(), 'processed_data', dataset)
user_id_index = pickle.load(open(os.path.join(processed_data_path, 'user_id_index.pkl'), 'rb'))
item_id_index = pickle.load(open(os.path.join(processed_data_path, 'item_id_index.pkl'), 'rb'))
data = np.loadtxt(os.path.join(processed_data_path, 'data.txt'), dtype=float)

# set split ratio
ratio = 0.6
train_data = data[:int(ratio*data.shape[0])]
vali_data = data[int(ratio*data.shape[0]):int((ratio+(1-ratio)/2)*data.shape[0])]
test_data = data[int((ratio+(1-ratio)/2)*data.shape[0]):]

NUM_USERS = max(user_id_index.values()) + 1
NUM_ITEMS = max(item_id_index.values()) + 1
print('dataset density:{:f}'.format(len(data)*1.0/(NUM_USERS*NUM_ITEMS)))


R = np.zeros([NUM_USERS, NUM_ITEMS])
for ele in train_data:
    R[int(ele[0]), int(ele[1])] = float(ele[2])

# construct model
print('training model.......')
lambda_alpha = 0.01
lambda_beta = 0.01
latent_size = 20
lr = 3e-5
iters = 1000
model = PMF(R=R, lambda_alpha=lambda_alpha, lambda_beta=lambda_beta, latent_size=latent_size, momuntum=0.9, lr=lr, iters=iters, seed=1)
print('parameters are:ratio={:f}, reg_u={:f}, reg_v={:f}, latent_size={:d}, lr={:f}, iters={:d}'.format(ratio, lambda_alpha, lambda_beta, latent_size,lr, iters))
U, V, train_loss_list, vali_rmse_list = model.train(train_data=train_data, vali_data=vali_data)

print('testing model.......')
preds = model.predict(data=test_data)
test_rmse = RMSE(preds, test_data[:, 2])

print('test rmse:{:f}'.format(test_rmse))
