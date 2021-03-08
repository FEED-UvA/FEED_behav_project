import sys
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

sys.path.append('src')
from utils import argmax_with_random_ties
from noise_ceiling import convert_doubles_to_single_labels


ohe = OneHotEncoder(
        categories='auto',
        sparse=False
)
ohe.fit(np.array([0, 1, 2, 3, 4, 5])[:, np.newaxis])


def cross_val_predict_and_score(estimator, X, y, cv, scoring, between_sub=True, soft=True):
    """ Cross-validation function which also keeps track of coefficients.

    X : DataFrame
        Pandas df with predictors
    y : Series
        Pandas series with target
    cv : cv object
        Sklearn cross-validator object
    scoring : func
        Scoring function
    """

    if isinstance(y, pd.DataFrame):
        K = y.shape[1]
        classes = y.columns.tolist()
    else:
        K = y.unique().size
        classes = np.sort(y.unique())
    
    N = y.shape[0]

    n_reps = 1 if not hasattr(cv, 'n_repeats') else cv.n_repeats
    n_splits = cv.get_n_splits() // n_reps

    # Split and pre-allocate DFs
    if between_sub:        
        # now, temporarily get hard labels, so that we can make
        # a stratified split
        y_hard = convert_doubles_to_single_labels(y, soft=False, keepdims=False)
        cv_gen = cv.split(np.random.normal(0, 1, size=(y_hard.shape[0], 1)), y_hard.values.argmax(axis=1))
    else:
        cv_gen = cv.split(X, y)
    
    # Pre-allocate predictions
    preds = [pd.DataFrame(np.zeros((N, K)), columns=classes, index=X.index)
             for _ in range(n_reps)]
    
    estimators = []  # to save for later
    i_rep = 0
    for i, (train_idx_int, test_idx_int) in enumerate(cv_gen):

        if between_sub:
            train_idx = y_hard.index[train_idx_int]
            test_idx = y_hard.index[test_idx_int]
        else:
            train_idx = y.index[train_idx_int]
            test_idx = y.index[test_idx_int]
        
        X_train = X.loc[train_idx, :]
        y_train = y.loc[train_idx]
        estimator.fit(X_train, y_train)
        
        X_test = X.loc[test_idx, :]  
        preds[i_rep].loc[test_idx, :] = estimator.predict_proba(X_test)
        estimators.append(estimator)

        if (i + 1) % n_splits == 0:
            i_rep += 1

    scores = np.zeros((n_reps, K))
    for i in range(n_reps):
        if not soft:
            these_preds = preds[i].values.argmax(axis=1)
            these_preds = ohe.transform(these_preds[:, np.newaxis])
        else:
            these_preds = preds[i].values

        y_ohe = ohe.transform(y.values[:, np.newaxis])
        scores[i, :] = scoring(y_ohe, these_preds, average=None)
    
    # Average across
    scores = scores.mean(axis=0)

    for i in range(n_reps):
        preds[i]['y_true'] = y.values

    # Get mean coefs
    coef = np.zeros((len(estimators), K, X.shape[1]))
    for i, est in enumerate(estimators):  # loop over folds
        if hasattr(est, 'steps'):  # is Pipeline
            est = est.steps[-1][1]
            if hasattr(est, 'best_estimator_'):  # is GridSearchCV
                est = est.best_estimator_

        # Store coefficients
        if hasattr(est, 'coef_'):
            coef[i, :, :] = est.coef_
    
    coef = np.mean(coef, axis=0)  # average coefs across folds
    return preds, scores, coef, est


def cross_val_predict_and_score_fullCV(estimator, X, y, cv, scoring, per_class=True, X_val=None, y_val=None,
                                return_model=True):
    """ Cross-validation function which also keeps track of coefficients.

    X : DataFrame
        Pandas df with predictors
    y : Series
        Pandas series with target
    cv : cv object
        Sklearn cross-validator object
    scoring : func
        Scoring function
    per_class : bool
        Whether to score per class
    X_val : DataFrame
        Optional validation targets
    y_val : DataFrame
        Optional validation predictors 
    """

    K = y.unique().size
    classes = np.sort(y.unique())
    N = y.size

    has_val = True if X_val is not None and y_val is not None else False
    n_reps = 1 if not hasattr(cv, 'n_repeats') else cv.n_repeats
    n_splits = cv.get_n_splits() // n_reps

    # Split and pre-allocate DFs
    if has_val:
        cv_gen = cv.split(X_val, y_val)
        N = X_val.shape[0]
        preds = [pd.DataFrame(np.zeros((N, K)), columns=classes, index=X_val.index)
                 for _ in range(n_reps)]
    else:
        cv_gen = cv.split(X, y)
        preds = [pd.DataFrame(np.zeros((N, K)), columns=classes, index=X.index)
                 for _ in range(n_reps)]
    
    estimators = []  # to save for later
    i_rep = 0
    for i, (train_idx_int, test_idx_int) in enumerate(cv_gen):
        
        if has_val:
            train_idx = X.index.intersection(X_val.index[train_idx_int])
            test_idx = X.index.intersection(X_val.index[test_idx_int])
        else:
            # Get str idx
            train_idx = X.index[train_idx_int]
            test_idx = X.index[test_idx_int]

        # Always fit on X/y (never on X_val/y_val)
        estimator.fit(X.loc[train_idx, :], y.loc[train_idx])
        
        if has_val:
            test_idx = X_val.index[test_idx_int]
            to_pred = X_val.loc[test_idx, :]
        else:
            to_pred = X.loc[test_idx, :]
        
        preds[i_rep].loc[test_idx, :] = estimator.predict_proba(to_pred)        
        estimators.append(estimator)

        if (i + 1) % n_splits == 0:
            i_rep += 1

    y2use = y_val if has_val else y
    scores = np.zeros((n_reps, K))
    for i in range(n_reps):
        scores[i, :] = scoring(y2use.values, preds[i].values, per_class=per_class)
    scores = scores.mean(axis=0)

    # Get mean coefs
    coef = np.zeros((len(estimators), K, X.shape[1]))
    for i, est in enumerate(estimators):  # loop over folds
        if hasattr(est, 'steps'):  # is Pipeline
            est = est.steps[-1][1]
            if hasattr(est, 'best_estimator_'):  # is GridSearchCV
                est = est.best_estimator_

        # Store coefficients
        coef[i, :, :] = est.coef_
    
    coef = np.mean(coef, axis=0)  # average coefs across folds
    if return_model:
        est.coef_ = coef
        return scores, coef, est
    else:
        return scores, coef
