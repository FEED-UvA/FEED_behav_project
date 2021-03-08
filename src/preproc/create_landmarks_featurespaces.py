import itertools
import matplotlib
matplotlib.use("Agg")
import numpy as np
import os.path as op
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from glob import glob
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA


info = pd.read_csv('../stims/stimuli-expressive_selection-all.csv', sep='\t', index_col=0)
info = info.sort_index()

dirs = sorted(glob('../FEED_stimulus_frames/id-*'))
N = len(dirs)

index = [op.basename(d) for d in dirs]
if not op.isfile('data/landmarks_3D.npy'):
    lms = np.zeros((N, 30, 50, 3))  # stims, frames, landmarks, coords (x, y, z)
    for i, d in enumerate(tqdm(dirs)):
        files = sorted(glob(d + '/*/texmap/*landmarks.mat'))
        for ii, f in enumerate(files):
            lms[i, ii, :, :] = loadmat(f)['landmarks']

    np.save('data/landmarks_3D.npy', lms)
    np.savetxt('data/stim_names.txt', index, fmt="%s")
else:
    lms = np.load('data/landmarks_3D.npy')

### Save raw (3D) landmarks
colnames = [f'lm-{i+1}_coord-{c}' for i in range(50) for c in ['x', 'y', 'z']]
df_f01 = pd.DataFrame(lms[:, 0, :, :].reshape((N, 50 * 3)), columns=colnames, index=index)
#df_f01['data_split'] = info.loc[df_f01.index, 'data_split']
df_f01.to_csv('data/featurespace-lm_frame-01.tsv', sep='\t')

df_f15 = pd.DataFrame(lms[:, 14, :, :].reshape((N, 50 * 3)), columns=colnames, index=index)
#df_f15['data_split'] = info.loc[df_f15.index, 'data_split']
df_f15.to_csv('data/featurespace-lm_frame-15.tsv', sep='\t')

df_f15min01 = pd.DataFrame(df_f15.values - df_f01.values, columns=colnames, index=index)
#df_f15min01['data_split'] = info.loc[df_f15min01.index, 'data_split']
df_f15min01.to_csv('data/featurespace-lm_frame-15min01.tsv', sep='\t')

# MOTION!
motion = np.sqrt(np.sum(np.diff(lms, axis=1) ** 2, axis=1)).reshape((N, 50 * 3))
df_mot = pd.DataFrame(motion, columns=colnames, index=index)
df_mot.to_csv('data/featurespace-lmmotion.tsv', sep='\t')

### Save PCA-reduced raw (3D) landmarks
N_COMPS = 50
colnames = [f'comp_{str(i+1).zfill(3)}' for i in range(N_COMPS)]
pca = PCA(n_components=N_COMPS)

for norm in [False, True]:

    if norm:
        # lms: stim x frames x landmarks (50) x coord (x, y, z)
        nz = ~np.isclose(lms.std(axis=(0, 1)), 0.)
        lms[..., nz] -= lms[..., nz].mean(axis=(0, 1))
        lms[..., nz] /= lms[..., nz].std(axis=(0, 1))

    ns = 'y' if norm else 'n'

    pca.fit(df_f01.values)
    np.savez(f'models/featurespace-lmPCA_frame-01_norm-{ns}_weights.npz', mu=pca.mean_, w=pca.components_)
        
    df_f01 = pd.DataFrame(pca.transform(df_f01.values), columns=colnames, index=index)
    #df_f01['data_split'] = info.loc[df_f01.index, 'data_split']
    df_f01.to_csv(f'data/featurespace-lmPCA_frame-01_norm-{ns}.tsv', sep='\t')

    pca.fit(df_f15.values)
    np.savez(f'models/featurespace-lmPCA_frame-15_norm-{ns}_weights.npz', mu=pca.mean_, w=pca.components_)

    df_f15 = pd.DataFrame(pca.transform(df_f15.values), columns=colnames, index=index)
    #df_f15['data_split'] = info.loc[df_f15.index, 'data_split']
    df_f15.to_csv(f'data/featurespace-lmPCA_frame-15_norm-{ns}.tsv', sep='\t')

    pca.fit(lms.reshape((N * 30, 50 * 3)))  # fit on ALL frames and stimuli
    np.savez(f'models/featurespace-lmPCA_frame-all_norm-{ns}_weights.npz', mu=pca.mean_, w=pca.components_)

    change = pca.transform(lms[:, 14, :, :].reshape((N, 50 * 3))) - pca.transform(lms[:, 0, :, :].reshape((N, 50 * 3)))
    df_f15min01 = pd.DataFrame(change, columns=colnames, index=index)
    #df_f15min01['data_split'] = info.loc[df_f15min01.index, 'data_split']
    df_f15min01.to_csv(f'data/featurespace-lmPCA_frame-15min01_norm-{ns}.tsv', sep='\t')

    diff = (lms[:, 14, :, :] - lms[:, 0, :, :]).reshape((N, 50 * 3))
    pca.fit(diff)
    np.savez(f'models/featurespace-lmPCA_frame-diff_norm-{ns}_weights.npz', mu=pca.mean_, w=pca.components_)
    df_f15min01 = pd.DataFrame(pca.transform(diff), columns=colnames, index=index)
    #df_f15min01['data_split'] = info.loc[df_f15min01.index, 'data_split']
    df_f15min01.to_csv(f'data/featurespace-lmPCA_frame-diff_norm-{ns}.tsv', sep='\t')
    
    ### Motion magnitude based
    pca.fit(motion)
    df_motion = pd.DataFrame(pca.transform(motion), columns=colnames, index=index)
    #df_mag['data_split'] = info.loc[df_mag.index, 'data_split']
    df_motion.to_csv('data/featurespace-lmmotionPCA.tsv', sep='\t')

### Save landmarks distances (3D)
f_out = 'data/landmarks_distances.npy'
combs = list(itertools.combinations(range(1, 51), 2))
colnames = [f'lm{i}-lm{j}' for i, j in combs]

if not op.isfile(f_out):
    
    lmdist = np.zeros((N, 30, len(combs)))
    triu_idx = np.triu_indices(50, k=1)
    for i in range(lms.shape[0]):  # stims
        for ii in range(lms.shape[1]):  # frames
            lmdist[i, ii, :] = pairwise_distances(lms[i, ii, :, :])[triu_idx]
    np.save(f_out, lmdist)
else:
    lmdist = np.load(f_out)

df_f01 = pd.DataFrame(lmdist[:, 0, :], columns=colnames, index=index)
#df_f01['data_split'] = info.loc[df_f01.index, 'data_split']
df_f01.to_csv('data/featurespace-lmdist_frame-01.tsv', sep='\t')

df_f15 = pd.DataFrame(lmdist[:, 14, :], columns=colnames, index=index)
#df_f15['data_split'] = info.loc[df_f15.index, 'data_split']
df_f15.to_csv('data/featurespace-lmdist_frame-15.tsv', sep='\t')

df_f15min01 = pd.DataFrame(df_f15.values - df_f01.values, columns=colnames, index=index)
#df_f15min01['data_split'] = info.loc[df_f15min01.index, 'data_split']
df_f15min01.to_csv('data/featurespace-lmdist_frame-15min01.tsv', sep='\t')

### Save PCA-reduced landmark distances (3D)
colnames = [f'comp_{str(i+1).zfill(3)}' for i in range(N_COMPS)]

for norm in [False, True]:

    if norm:
        nz = ~np.isclose(lmdist.std(axis=(0, 1)), 0.)
        lmdist[:, :, nz] -= lmdist[:, :, nz].mean(axis=(0, 1))
        lmdist[:, :, nz] /= lmdist[:, :, nz].std(axis=(0, 1))

    ns = 'y' if norm else 'n'

    pca.fit(df_f01.values)
    np.savez(f'models/featurespace-lmdistPCA_frame-01_norm-{ns}_weights.npz', mu=pca.mean_, w=pca.components_)

    df_f01 = pd.DataFrame(pca.transform(df_f01.values), columns=colnames, index=index)
    #df_f01['data_split'] = info.loc[df_f01.index, 'data_split']
    df_f01.to_csv(f'data/featurespace-lmdistPCA_frame-01_norm-{ns}.tsv', sep='\t')

    pca.fit(df_f15.values)
    np.savez(f'models/featurespace-lmdistPCA_frame-15_norm-{ns}_weights.npz', mu=pca.mean_, w=pca.components_)

    df_f15 = pd.DataFrame(pca.transform(df_f15.values), columns=colnames, index=index)
    #df_f15['data_split'] = info.loc[df_f15.index, 'data_split']
    df_f15.to_csv(f'data/featurespace-lmdistPCA_frame-15_norm-{ns}.tsv', sep='\t')

    pca.fit(lmdist.reshape((N * 30, len(combs))))  # fit on ALL frames and stimuli
    np.savez(f'models/featurespace-lmdistPCA_frame-all_norm-{ns}_weights.npz', mu=pca.mean_, w=pca.components_)

    change = pca.transform(lmdist[:, 14, :]) - pca.transform(lmdist[:, 0, :])
    df_f15min01 = pd.DataFrame(change, columns=colnames, index=index)
    #df_f15min01['data_split'] = info.loc[df_f15min01.index, 'data_split']
    df_f15min01.to_csv(f'data/featurespace-lmdistPCA_frame-15min01_norm-{ns}.tsv', sep='\t')

    diff = lmdist[:, 14, :] - lmdist[:, 0, :]
    pca.fit(diff)
    np.savez(f'models/featurespace-lmdistPCA_frame-diff_norm-{ns}_weights.npz', mu=pca.mean_, w=pca.components_)

    change = pca.transform(diff)
    df_f15min01 = pd.DataFrame(change, columns=colnames, index=index)
    #df_f15min01['data_split'] = info.loc[df_f15min01.index, 'data_split']
    df_f15min01.to_csv(f'data/featurespace-lmdistPCA_frame-diff_norm-{ns}.tsv', sep='\t')