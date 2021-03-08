import numpy as np
import pandas as pd
import os.path as op
from glob import glob


DU2EN = {
    'Verrassing': 'surprise',
    'Verdrietig': 'sadness',
    'Bang': 'fear',
    'Walging': 'disgust',
    'Blij': 'happiness',
    'Boos': 'anger',
    'Geen van allen': 'none'
}

tsvs = sorted(glob('data/ratings_complete/*expressive_ratings.tsv'))
dfs_evsn = []
dfs_emo = []
for tsv in tsvs:
    sub = op.basename(tsv).split("_")[0]
    print(f"Processing {sub} ...")
    df = pd.read_csv(tsv, sep='\t', index_col=0).query("rating_type == 'emotion'")
    y = df.loc[:, ['rating']]
    y['rating'] = [DU2EN[s] for s in y['rating']]
    uniq_idx = y.index.unique()
    emos = sorted(y['rating'].unique())
    counts = pd.DataFrame(np.zeros((uniq_idx.size, 7)), columns=emos, index=uniq_idx)

    for i, idx in enumerate(uniq_idx):
        this = y.loc[idx, 'rating']
        if isinstance(this, str):
            this = [this]
        
        for r in this:
            counts.loc[idx, r] = counts.loc[idx, r] + 1
    
    # Pre-allocate best prediction array
    labels = np.zeros_like(counts, dtype=float)

    for ii in range(counts.shape[0]):
        # Determine most frequent label across reps
        opt_class = np.where(counts.iloc[ii, :] == counts.iloc[ii, :].max())[0]
        rnd_class = np.random.choice(opt_class, size=1)  
        labels[ii, rnd_class] = 1

    # Repeat best possible prediction R times
    labels = pd.DataFrame(labels, index=uniq_idx, columns=emos)
    emo_vs_none = (1 - labels[['none']]).rename({'none': 'emo'}, axis=1).sort_index()
    emo_vs_none['sub'] = sub
    dfs_evsn.append(emo_vs_none)

    emo = labels.drop('none', axis=1)
    emo['sub'] = sub
    dfs_emo.append(emo)
    # TODO: add data_split

df_evsn = pd.concat(dfs_evsn, axis=0)
df_evsn.to_csv('data/featurespace-emoVSnone.tsv', sep='\t')

df_emo = pd.concat(dfs_emo, axis=0)
df_emo.to_csv('data/featurespace-categoricalemotion.tsv', sep='\t')
