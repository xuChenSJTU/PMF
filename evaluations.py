import numpy as np

def RMSE(preds, truth):
    return np.sqrt(np.mean(np.square(preds-truth)))