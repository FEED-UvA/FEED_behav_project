import azure
import pickle
import os.path as op
import numpy as np
import pandas as pd
from glob import glob
from azure.cognitiveservices.vision.face.models._models_py3 import Coordinate

def load_landmarks(stim, api='google', frames_dir='../FEED_stimulus_frames'):

    stims = sorted(glob(f'{frames_dir}/{stim}/*/texmap/frame*.png'))
    
    # number of landmarks: 27 (azure), 34 (google)
    n_lm = 27 if api == 'azure' else 34
    df = pd.DataFrame(columns=['frame', 'landmark', 'coords'], index=range(n_lm * 30))

    row = 0
    for i in range(30):  # loop across frames
        info = stims[i].replace('.png', f'_api-{api}_annotations.pkl')
        with open(info, 'rb') as f_in:
            info = pickle.load(f_in)
        
        if api == 'azure':
            info = info[0].face_landmarks
            ii = 0
            for attr in dir(info):
                this_attr = getattr(info, attr)                
                if isinstance(this_attr, Coordinate):
                    df.loc[row, 'frame'] = i
                    df.loc[row, 'landmark'] = ii
                    df.loc[row, 'coords'] = (this_attr.x, this_attr.y)
                    ii += 1
                    row += 1

        elif api == 'google':
            info = info.face_annotations[0]
            for ii in range(len(info.landmarks)):
                df.loc[row, 'frame'] = i
                df.loc[row, 'landmark'] = ii
                df.loc[row, 'coords'] = (
                    np.round(info.landmarks[ii].position.x, 1),
                    np.round(info.landmarks[ii].position.y, 1)
                )
                row += 1
        else:
            raise ValueError("Choose api from 'google' and 'azure'.")

    df['stim'] = stim                
    df['api'] = api
    df = df.loc[:, ['stim', 'frame', 'landmark', 'coords', 'api']]
    return df


if __name__ == '__main__':

    from tqdm import tqdm
    from joblib import Parallel, delayed

    stims = [op.basename(d) for d in sorted(glob('../FEED_stimulus_frames/id*'))]
    dfs = Parallel(n_jobs=15)(delayed(load_landmarks)(s, api)
        for s in tqdm(stims) for api in ['azure', 'google']
    )
    df = pd.concat(dfs, axis=0).set_index('stim')
    df.to_csv('data/landmarks.tsv', sep='\t')