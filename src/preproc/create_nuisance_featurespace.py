import os.path as op
import pandas as pd
import numpy as np
from glob import glob

df_generic = pd.read_csv('../stims/stimuli-expressive_selection-train+test.tsv', sep='\t', index_col=0)

LAGS = 2
to_keep = ['rep', 'data_split', 'session', 'run', 'block', 'trial', 'rating_RT', 'rating_type']
tsvs = sorted(glob('data/ratings_complete/*expressive_ratings.tsv'))
dfs = []
for tsv in tsvs:
    sub = op.basename(tsv).split("_")[0]
    print(f"Processing {sub} ...")
    
    df = pd.read_csv(tsv, sep='\t', index_col=0).loc[:, to_keep + ['rating']]
    df = df.query("rating_type == 'emotion'").drop('rating_type', axis=1)
    orig_idx = df.index.copy()
    df = df.reset_index()
    
    """
    for rep in [1, 2, 3]:
        
        #for run in range(1, 17):
        for ses in [1, 2, 3]:
            
            if ses > 1 and rep > 1:
                continue
            
            for block in np.arange(1, 17):
                tmp = df.query("rep == @rep & session == @ses & block == @block").copy()
                
                #tmp = df.query("rep == @rep & session == @ses & block == @block & run == @run").copy()
                tmp['rating_t_min_1'] = np.r_[np.nan, tmp['rating'].values[:-1]]
                df.loc[tmp.index, 'rating_t_min_1'] = tmp.loc[:, 'rating_t_min_1']
                for lag in np.arange(2, LAGS+1):
                    tmp.loc[:, f'rating_t_min_{lag}'] = np.r_[np.nan, tmp[f'rating_t_min_{lag-1}'].iloc[:-1]]
                    df.loc[tmp.index, f'rating_t_min_{lag}'] = tmp.loc[:, f'rating_t_min_{lag}']
    """
    df.index = orig_idx
    df['sub'] = sub
    df = df.loc[:, ['sub', 'data_split'] + [col for col in df.columns if col not in ['sub', 'data_split']]]
    df = df.set_index('index')
    df = df.sort_values(by=['session', 'run', 'block', 'trial', 'rep'], axis=0)
    dfs.append(df.drop(['rating'], axis=1))

df = pd.concat(dfs, axis=0)
df.to_csv('data/featurespace-nuisance.tsv', sep='\t')
print(df.shape)