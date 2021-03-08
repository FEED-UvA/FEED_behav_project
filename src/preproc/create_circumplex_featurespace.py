import os.path as op
import pandas as pd
import numpy as np
from glob import glob

df_generic = pd.read_csv('../stims/stimuli-expressive_selection-train+test.tsv', sep='\t', index_col=0)

tsvs = sorted(glob('data/ratings_complete/*expressive_ratings.tsv'))
dfs = []
for tsv in tsvs:
    sub = op.basename(tsv).split("_")[0]
    print(f"Processing {sub} ...")
    
    df = pd.read_csv(tsv, sep='\t', index_col=0).loc[:, ['data_split', 'rep', 'rating', 'rating_type', 'trial_type']]
    df = df.query("rating_type != 'emotion'")
    df_orig = df.copy()
    df['rating'] = df['rating'].astype(float)
    df = df.groupby(['trial_type', 'rating_type']).mean()['rating'].reset_index()
    df = df.pivot(index='trial_type', columns='rating_type', values='rating')
    df['sub'] = sub
    df['data_split'] = df_generic.loc[df.index, 'data_split']
    df = df.loc[:, ['sub', 'data_split'] + [col for col in df.columns[:-1] if col != 'sub']]
    dfs.append(df)

df = pd.concat(dfs, axis=0)#.set_index('trial_type')
df.to_csv('data/featurespace-circumplex.tsv', sep='\t')
print(df.shape)