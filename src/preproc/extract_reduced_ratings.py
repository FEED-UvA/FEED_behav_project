import sys
import numpy as np
import os.path as op
import pandas as pd
from glob import glob
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

sys.path.append('src')
from noise_ceiling import convert_doubles_to_single_labels


DU2EN = dict(
    Verrassing='surprise',
    Verdrietig='sadness',
    Bang='fear',
    Walging='disgust',
    Blij='happiness',
    Boos='anger'
)

le = LabelEncoder().fit(['happiness', 'surprise', 'fear', 'sadness', 'disgust', 'anger'])

paths = sorted(glob('data/ratings_complete/*expressive*'))
for path in tqdm(paths):
    df = pd.read_csv(path, sep='\t', index_col=0)
    #df = df.query("rating_type == @target")  # filter rating type
    df = df.query("rating != 'Geen van allen'")

    with pd.option_context('mode.chained_assignment', None):  # suppress stupid warning
        df.loc[:, 'rating'] = df.loc[:, 'rating'].replace(DU2EN)  # translate
        df_emo = df.query("rating_type == 'emotion'")
        df_emo.loc[:, 'rating'] = le.transform(df_emo['rating'])

    df_emo = convert_doubles_to_single_labels(df_emo['rating'], soft=False, keepdims=False)
    df_emo = pd.DataFrame(df_emo.values.argmax(axis=1), columns=['rating'], index=df_emo.index)
    df_emo['rating'] = le.inverse_transform(df_emo['rating'])
    df_emo = df_emo.rename({'rating': 'emotion'}, axis=1)

    df_tmp = df.query("rating_type == 'valence'")
    df_val = pd.DataFrame()
    for idx in df_emo.index:
        tmp = df_tmp.loc[idx, 'rating']
        if isinstance(tmp, pd.Series):
            val = df_tmp.loc[idx, 'rating'].astype(float).values.mean()
        else:
            val = tmp

        df_val.loc[idx, 'valence'] = val

    df_tmp = df.query("rating_type == 'arousal'")
    df_aro = pd.DataFrame()
    for idx in df_emo.index:
        tmp = df_tmp.loc[idx, 'rating']
        if isinstance(tmp, pd.Series):
            val = df_tmp.loc[idx, 'rating'].astype(float).values.mean()
        else:
            val = tmp

        df_aro.loc[idx, 'arousal'] = val

    df_all = pd.concat((df_emo.sort_index(), df_val.sort_index(), df_aro.sort_index()), axis=1)
    df_all.to_csv(f'data/ratings_reduced/{op.basename(path)}', sep='\t')