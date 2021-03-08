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
from joblib import Parallel, delayed

sys.path.append(op.abspath(op.dirname(op.dirname(__file__))))
from data_io import DataLoader
from model import cross_val_predict_and_score
from metrics import roc_auc_score_per_class


def classify_subjects_parallel(sub, subs, feature_spaces, model, cv):
    """ Helper function to parallelize analysis across subjects. """
    scores, coefs = [], dict()
    for i, fs in enumerate(feature_spaces):
        
        if not isinstance(fs, (tuple, list)):
            fs = (fs,)
        
        fs_name = '+'.join(fs)

        dl = DataLoader(sub=sub, log_level=30)
        dl.load_y(strategy_doubles='hard')
        dl.load_X(feature_set=fs, n_comp=100)
        X_val, y_val = dl.return_Xy()
        
        other_X, other_y = [], []
        other_subs = [s for s in subs if s != sub]
        for other_sub in other_subs:
            dl = DataLoader(sub=other_sub, log_level=30)
            dl.load_y(strategy_doubles='hard')
            dl.load_X(feature_set=fs, n_comp=100)
            this_X, this_y = dl.return_Xy()
            other_X.append(this_X)
            other_y.append(this_y)
        
        X = pd.concat(other_X, axis=0)
        y = pd.concat(other_y, axis=0)

        scores_, coefs_, model_ = cross_val_predict_and_score(
            estimator=model,
            X=X, y=y,
            cv=cv,
            scoring=roc_auc_score_per_class,
            X_val=X_val,
            y_val=y_val,
            per_class=True,
            return_model=True
        )
        joblib.dump(model_, f'models/sub-{sub}_type-between_fs-{fs_name}_model.jl')

        dl.log.warning(f"sub-{sub} scores: {np.round(scores_, 2)} (fs = {fs_name})")
        scores_df = pd.DataFrame(scores_, columns=['score'])
        scores_df['feature_set'] = fs_name
        scores_df['emotion'] = dl.le.classes_
        scores_df['sub'] = sub
        scores.append(scores_df)

        coefs_df = pd.DataFrame(data=coefs_, columns=X.columns)
        coefs_df['feature_set'] = fs_name
        coefs_df['emotion'] = dl.le.classes_
        coefs_df['sub'] = sub
        coefs[fs_name] = coefs_df
        
    scores_df = pd.concat(scores, axis=0)    
    return scores_df, coefs
    

if __name__ == '__main__':

    subs = [str(s).zfill(2) for s in range(1, 14) if s != 11]
    feature_spaces = ['AU', 'nmf14min0', 'pca14min0', 'AUxAU', 'SJ', ['AU', 'SJ'], 'circumplex']
    
    # Define pipeline
    model = make_pipeline(
        StandardScaler(),
        GridSearchCV(
            estimator=LogisticRegression(
                penalty='l2',
                class_weight='balanced',
                n_jobs=1,
                solver='liblinear',
                max_iter=100,
            ),
            param_grid=dict(
                C=(0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000),
            ),
            cv=10,
            n_jobs=1
        )
        #LogisticRegression(penalty='l2', class_weight='balanced', n_jobs=1, solver='liblinear')
    )
    cv = RepeatedStratifiedKFold(n_repeats=2, n_splits=5)

    # Run parallel
    out = Parallel(n_jobs=13)(delayed(classify_subjects_parallel)
        (sub, subs, feature_spaces, model, cv) for sub in subs
    )

    # Concatenate scores across participants
    scores = pd.concat([o[0] for o in out], axis=0)
    coefs = [o[1] for o in out]

    # Save
    root_dir = op.dirname(op.dirname(op.dirname(__file__)))
    f_out = op.join(root_dir, 'results', 'analysis-betweensub_split-train_scores.tsv')
    scores.to_csv(f_out, sep='\t')

    for fs in feature_spaces:
        
        if not isinstance(fs, (tuple, list)):
            fs = (fs,)
        
        fs_name = '+'.join(fs)
        coefs_df = pd.concat([o[fs_name] for o in coefs], axis=0)    
        f_out = op.join(root_dir, 'results', f'analysis-betweensub_split-train_fs-{fs_name}_coefs.tsv')
        coefs_df.to_csv(f_out, sep='\t')
