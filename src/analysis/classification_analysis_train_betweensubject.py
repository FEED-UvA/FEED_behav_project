import sys
import joblib
import os.path as op
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, f1_score
from joblib import Parallel, delayed

sys.path.append(op.abspath(op.dirname(op.dirname(__file__))))
from data_io import DataLoader
from model import cross_val_predict_and_score
from metrics import tjur_score


def classify_fs_parallel(subs, fs, model, cv):
    """ Helper function to parallelize analysis across FSs. """
        
    if not isinstance(fs, (tuple, list)):
        fs = (fs,)
    
    fs_name = '+'.join(fs)

    X, y = [], []
    for sub in subs:
        dl = DataLoader(sub=sub, log_level=30)
        dl.load_y(strategy_doubles='hard')
        dl.load_X(feature_set=fs, n_comp=100)
        this_X, this_y = dl.return_Xy()
        X.append(this_X)
        y.append(this_y)
    
    X = pd.concat(X, axis=0)
    y = pd.concat(y, axis=0)
    
    preds_, scores_, coefs_, model_ = cross_val_predict_and_score(
        estimator=model,
        X=X, y=y,
        cv=cv,
        scoring=tjur_score,
        soft=True
    )

    joblib.dump(model_, f'models/sub-{sub}_analysis-between_split-train_fs-{fs_name}_model.jl')
    
    dl.log.warning(f"sub-{sub} scores: {np.round(scores_, 2)} (fs = {fs_name})")
    scores = pd.DataFrame(scores_, columns=['score'])
    scores['feature_set'] = fs_name
    scores['emotion'] = dl.le.classes_
    scores['sub'] = sub
    
    for i in range(len(preds_)):
        preds_[i]['feature_set'] = fs_name
        preds_[i]['rep'] = i
    
    preds = pd.concat(preds_, axis=0)

    coefs = pd.DataFrame(data=coefs_, columns=X.columns)
    coefs['feature_set'] = fs_name
    coefs['emotion'] = dl.le.classes_
    
    return preds, scores, coefs


if __name__ == '__main__':

    subs = [str(s).zfill(2) for s in range(1, 14) if s != 11]
    feature_spaces = ['lmdist_frame-15min01', 'pixelPCA_frame-15min01', 'AUxAU', 'SJ', ['AU', 'SJ'], 'AU']
    
    # Define pipeline
    model = make_pipeline(
        StandardScaler(),
        #GridSearchCV(
        #    estimator=LogisticRegression(
        #        penalty='l2',
        #        class_weight='balanced',
        #        n_jobs=1,
        #        solver='liblinear',
        #        max_iter=100,
        #    ),
        #    param_grid=dict(
        #        C=(0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000),
        #    ),
        #    cv=10,
        #    n_jobs=1
        #)
        LogisticRegression(penalty='l2', class_weight='balanced', n_jobs=1, solver='liblinear')
    )
    cv = RepeatedStratifiedKFold(n_repeats=1, n_splits=5)

    # Run parallel
    out = Parallel(n_jobs=12)(delayed(classify_fs_parallel)
        (subs, fs, model, cv) for fs in feature_spaces
    )

    # Concatenate scores across participants
    preds = pd.concat([o[0] for o in out], axis=0)
    scores = pd.concat([o[1] for o in out], axis=0)
    coefs = [o[2] for o in out]

    # Save
    root_dir = op.dirname(op.dirname(op.dirname(__file__)))
    f_out = op.join(root_dir, 'results', 'analysis-between_split-train_scores.tsv')
    scores.to_csv(f_out, sep='\t')
    preds.to_csv(f_out.replace('_scores', '_preds'), sep='\t')

    for coefs_df, fs in zip(coefs, feature_spaces):
        
        if not isinstance(fs, (tuple, list)):
            fs = (fs,)
        
        fs_name = '+'.join(fs)
        f_out = op.join(root_dir, 'results', f'analysis-between_split-train_fs-{fs_name}_coefs.tsv')
        coefs_df.to_csv(f_out, sep='\t')
