import pandas as pd
import os.path as op
from glob import glob
from ipdb import set_trace

df_generic = pd.read_csv('../stims/stimuli-expressive_selection-all.csv', sep='\t', index_col=0)

tsvs = sorted(glob('data/ratings_complete/*neutral_ratings.tsv'))
dfs = []
for tsv in tsvs:
    sub = op.basename(tsv).split("_")[0]
    print(f"Processing {sub} ...")
    df_generic = pd.read_csv('data/featurespace-AU.tsv', sep='\t', index_col=0)

    df = (
        pd.read_csv(tsv, sep='\t', index_col=0)
          .drop(['session', 'run', 'trial', 'block', 'rating_RT'], axis=1)
          .groupby(['trial_type', 'rating_type']).mean()['rating']
    )
    
    df = df.reset_index().pivot(index='trial_type', columns='rating_type', values='rating')
    df.index = [idx.split('_')[0] for idx in df.index]
    
    X_SA = df.loc[df_generic.index.str.split('_').str[0], :]
    X_SA = X_SA.set_index(df_generic.index)
    X_SA['sub'] = sub
    print(X_SA)
    #X_SA['data_split'] = df_generic.loc[X_SA.index, 'data_split'].copy()
    X_SA = X_SA.loc[:, ['sub', 'data_split'] + [col for col in X_SA.columns[:-1] if col != 'sub']]
    dfs.append(X_SA)

df = pd.concat(dfs, axis=0)
df.to_csv('data/featurespace-SJ.tsv', sep='\t')
