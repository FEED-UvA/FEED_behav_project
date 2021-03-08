import sys
import numpy as np
import os.path as op
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, recall_score

sys.path.append(op.abspath(op.dirname(op.dirname(__file__))))
from data_io import DataLoader
from noise_ceiling import compute_noise_ceiling
from metrics import brier_score, tjur_score


subs = [str(s).zfill(2) for s in range(1, 14) if s != 11]
ceilings = np.zeros((len(subs), 6))
y_all = []
for i, sub in enumerate(subs):
    dl = DataLoader(sub=sub, log_level=30)
    y_doubles = dl.load_y(return_doubles=True)
    ceilings[i, :] = compute_noise_ceiling(y_doubles, soft=True, scoring=tjur_score)
    dl.log.warning(f"Ceiling sub-{sub}: {ceilings[i, :]}")

    # Note to self: between-subject NC only works with 'hard' labels,
    # otherwise you need to deal with two sources of "doubles"/inconsistency
    dl.load_y(return_doubles=False, strategy_doubles='hard')
    y_all.append(dl.y)

# Ceilings per subject
ceilings = pd.DataFrame(ceilings, columns=dl.le.classes_, index=subs)

# Ceiling across subjects
y = pd.concat(y_all, axis=0)
pd.get_dummies(y).to_csv('results/y_all.tsv', sep='\t')

ceilings.loc['between', :] = compute_noise_ceiling(y, soft=True, scoring=tjur_score, doubles_only=True)
dl.log.warning(f"Ceiling between-sub: {ceilings.loc['between', :].values}")
ceilings.to_csv('results/noise_ceilings.tsv', sep='\t')