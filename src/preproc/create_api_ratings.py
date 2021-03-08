import pickle
import numpy as np
import pandas as pd
import os.path as op
from glob import glob

frames_dir = op.join('..', 'FEED_stimulus_frames')

labels = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
GO2IDX = dict(sorrow=4, joy=3, surprise=1, anger=0)
for api in ['google', 'azure']:

    files = sorted(glob(op.join(frames_dir, 'id-*', '*frames', 'texmap', f'frame14_api-{api}_annotations.pkl')))
    if not files:
        raise ValueError("Could not find files.")

    ratings = np.zeros((len(files), 6))
    for i, f in enumerate(files):
        with open(f, 'rb') as f_in:
            info = pickle.load(f_in)
            if api == 'google':
                info = info.face_annotations[0]
                for emo in ['sorrow', 'surprise', 'joy', 'anger']:
                    emo_ll = getattr(info, f'{emo}_likelihood')
                    ratings[i, GO2IDX[emo]] = emo_ll / 5
            else:
                info = info[0]
                for lab in labels:
                    emo_ll = getattr(info.face_attributes.emotion, lab)
                    ratings[i, labels.index(lab)] = emo_ll
    
    #ratings_final = np.zeros_like(ratings)
    #ratings_final[np.arange(ratings_final.shape[0], ratings.argmax(axis=1))] = 1
    index = [op.basename(op.dirname(op.dirname(op.dirname(f)))) for f in files]
    df = pd.DataFrame(ratings, columns=labels, index=index)
    df.to_csv(f'data/api-{api}_emoratings.tsv', sep='\t')