import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score as f1_score_sklearn


def f1_score(y_true, y_pred, average=None):

    if y_true.ndim != 1:
        y_true = y_true.argmax(axis=1)

    if y_pred.ndim != 1:
        y_pred = y_pred.argmax(axis=1)

    return f1_score_sklearn(y_true, y_pred, average=average)


def brier_score(y_true, y_pred, scale=True, average=None):
    
    if average is None:
        brier = ((y_pred - y_true) ** 2).mean(axis=0)
    else:    
        brier = np.sum((y_pred - y_true) ** 2, axis=1).mean()
    
    if scale:
        if average is None:
            brier_max = np.mean(y_pred, axis=0) * np.mean(1 - y_pred, axis=0)
        else:
            brier_max = np.sum(np.mean(y_pred, axis=0) * np.mean(1 - y_pred, axis=0))
        
        brier = 1 - (brier / brier_max)
        
    return brier


def tjur_score(y_true, y_pred, average=False):

    k = y_true.shape[1]
    score_per_class = np.zeros(k)

    for i in range(k):
        score_0 = y_pred[y_true[:, i] == 0, i].mean()
        score_1 = y_pred[y_true[:, i] == 1, i].mean()
        score_per_class[i] = score_1 - score_0
    
    if average is None:
        return score_per_class
    else:
        return score_per_class.mean()
