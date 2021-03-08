import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

sys.path.append('src')
from data_io import DataLoader
from metrics import tjur_score

ohe = OneHotEncoder(categories='auto', sparse=False)
ohe.fit(np.arange(6)[:, np.newaxis])

scores_all = []
for api in ['google', 'azure']:
    df = pd.read_csv(f'data/api-{api}_emoratings.tsv', sep='\t', index_col=0)
    subs = [str(s).zfill(2) for s in range(1, 14) if s != 11]
    scores = np.zeros((len(subs), 6))
    for i, sub in enumerate(subs):
        dl = DataLoader(sub=sub, log_level=30)
        dl.load_y(strategy_doubles='hard')
        y_api = df.loc[dl.y.index].values
        y_true = ohe.transform(dl.y.values[:, np.newaxis])
        scores[i, :] = tjur_score(y_true, y_api, average=None)
        
    scores = pd.DataFrame(scores, columns=dl.le.classes_, index=subs).reset_index()
    scores = pd.melt(scores, id_vars='index', value_name='score', var_name='emotion')
    scores = scores.rename({'index': 'sub'}, axis=1)
    scores['api'] = api
    scores_all.append(scores)

scores = pd.concat(scores_all, axis=0)
scores.to_csv('results/api_ratings_scores.tsv', sep='\t')
print(scores.groupby(['api', 'emotion']).mean())
