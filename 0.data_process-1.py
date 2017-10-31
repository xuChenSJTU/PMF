import os
import pickle
import numpy as np
import pandas as pd

# choose dataset to process
dataset = 'ml-1m'
raw_data_path = os.path.join(os.getcwd(), 'data', dataset, 'ratings.dat')
processed_data_path = os.path.join(os.getcwd(), 'processed_data', dataset)

names = ['user_id', 'item_id', 'rating', 'timestamp']
read = pd.read_csv(raw_data_path, sep='::', names=names)


users = list(set(read['user_id']))
user_id_index = dict((user_id, index) for user_id, index in zip(users, range(len(users))))
items = list(set(read['item_id']))
item_id_index = dict((item_id, index) for item_id, index in zip(items, range(len(items))))

data = []
with open(os.getcwd()+'/data/'+dataset+'/ratings.dat', 'r') as f:
    lines = f.readlines()
    count_user = 0
    count_item = 0
    for i in range(len(lines)):
        line = lines[i].strip().split('::')
        user_id = int(line[0])
        item_id = int(line[1])
        rating = float(line[2])

        data.append([user_id_index[user_id], item_id_index[item_id], rating])

data = np.array(data)
np.random.shuffle(data)

pickle.dump(user_id_index, open(os.path.join(processed_data_path, 'user_id_index.pkl'), 'wb'))
pickle.dump(item_id_index, open(os.path.join(processed_data_path, 'item_id_index.pkl'), 'wb'))
np.savetxt(os.path.join(processed_data_path, 'data.txt'), data, fmt='%f')
