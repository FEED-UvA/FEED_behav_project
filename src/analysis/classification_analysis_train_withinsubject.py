import sys
import joblib
import os.path as op
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, roc_auc_score, recall_score, balanced_accuracy_score
from joblib import Parallel, delayed

sys.path.append(op.abspath(op.dirname(op.dirname(__file__))))
from data_io import DataLoader
from model import cross_val_predict_and_score
from metrics import tjur_score


def classify_subjects_parallel(sub, feature_spaces, model, cv):
    """ Helper function to parallelize analysis across subjects. """
    scores, preds, coefs = [], [], dict()
    for fs in feature_spaces:
        
        if not isinstance(fs, (tuple, list)):
            fs = (fs,)
        
        fs_name = '+'.join(fs)

        dl = DataLoader(sub=sub, log_level=30)
        dl.load_y(strategy_doubles='hard')
        dl.load_X(feature_set=fs, n_comp=100)
        X, y = dl.return_Xy()

        preds_, scores_, coefs_, model_ = cross_val_predict_and_score(
            estimator=model,
            X=X, y=y,
            cv=cv,
            scoring=tjur_score,
            between_sub=False,
            soft=True
        )
        joblib.dump(model_, f'models/sub-{sub}_analysis-within_split-train_fs-{fs_name}_model.jl')

        dl.log.warning(f"sub-{sub} scores: {np.round(scores_, 2)} (fs = {fs_name})")
        scores_df = pd.DataFrame(scores_, columns=['score'])
        scores_df['feature_set'] = fs_name
        scores_df['emotion'] = dl.le.classes_
        scores_df['sub'] = sub
        scores.append(scores_df)

        for i in range(len(preds_)):
            preds_[i]['feature_set'] = fs_name
            preds_[i]['sub'] = sub
            preds_[i]['rep'] = i
        
        preds.append(pd.concat(preds_, axis=0))

        coefs_df = pd.DataFrame(data=coefs_, columns=X.columns)
        coefs_df['feature_set'] = fs_name
        coefs_df['emotion'] = dl.le.classes_
        coefs_df['sub'] = sub
        coefs[fs_name] = coefs_df

    scores = pd.concat(scores, axis=0)
    preds = pd.concat(preds, axis=0)
    return preds, scores, coefs


if __name__ == '__main__':
    
    subs = [str(s).zfill(2) for s in range(1, 14) if s != 11]
    feature_spaces = [
        'lm_frame-01', 'lm_frame-15', 'lm_frame-15min01', 
        ['lm_frame-15min01', 'lm_frame-01']
    ]
     
    # Define pipeline
    model = make_pipeline(
        #GridSearchCV(
        #    estimator=LogisticRegression(
        #        penalty='l2',
        #        class_weight='balanced',
        #        n_jobs=1,
        #        solver='liblinear',
        #        max_iter=1000
        #    ),
        #    param_grid=dict(
        #        C=(0.001, 0.01, 0.1, 1, 10, 100),
        #    ),
        #    cv=10,
        #    n_jobs=1
        #)
        LogisticRegression(class_weight='balanced', n_jobs=1, C=1, max_iter=1000, solver='liblinear')
    )
    cv = RepeatedStratifiedKFold(n_repeats=1, n_splits=10)

    # Run parallel
    out = Parallel(n_jobs=13)(delayed(classify_subjects_parallel)
        (sub, feature_spaces, model, cv) for sub in subs
    )

    # Concatenate scores across participants
    preds = pd.concat([o[0] for o in out], axis=0)
    scores = pd.concat([o[1] for o in out], axis=0)
    print(scores.groupby('feature_set').mean())
    exit()
    coefs = [o[2] for o in out]

    # Save
    root_dir = op.dirname(op.dirname(op.dirname(__file__)))
    f_out = op.join(root_dir, 'results', 'analysis-within_split-train_scores.tsv')
    scores.to_csv(f_out, sep='\t')
    preds.to_csv(f_out.replace('_scores', '_preds'), sep='\t')

    for fs in feature_spaces:
        
        if not isinstance(fs, (tuple, list)):
            fs = (fs,)

        fs_name = '+'.join(fs)
        coefs_df = pd.concat([o[fs_name] for o in coefs], axis=0)    
        f_out = op.join(root_dir, 'results', f'analysis-within_split-train_fs-{fs_name}_coefs.tsv')
        coefs_df.to_csv(f_out, sep='\t')
