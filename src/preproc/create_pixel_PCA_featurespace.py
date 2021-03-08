import os.path as op
import imageio
from skimage.transform import rescale
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.decomposition import NMF, PCA
from joblib import Parallel, delayed


def _load_dir(d, scale=0.1):

    frames = sorted(glob(f'{d}/*_frames/texmap/frame*.png'))
    imgdata = np.zeros((30, int((800 * scale) * (600 * scale))))
    
    for i, frame in enumerate(frames):
        # Note to self: why average across RGB (mean(axis=-1))?
        img = imageio.imread(frame)[:, :, :-1].mean(axis=-1)
        imgdata[i, :] = rescale(img, scale).ravel()

    return imgdata


if __name__ == '__main__':    

    SCALE = 0.25
    N_COMPS = 50
    N_JOBS = 10

    info = pd.read_csv('../stims/stimuli-expressive_selection-train+test.tsv', sep='\t', index_col=0)

    dirs = sorted(glob(f'../FEED_stimulus_frames/*'))
    dirs = [d for d in dirs if op.basename(d) in info.index.tolist()]
    #imgdata = np.zeros((len(dirs), 30, int((800 * SCALE) * (600 * SCALE))))
    index = [op.basename(d) for d in dirs]
    
    f_out = f'data/scale-{int(SCALE * 100)}_pixels.npy'
    if not op.isfile(f_out):
        print("Loading image data ...")
        imgdata = Parallel(n_jobs=N_JOBS)(delayed(_load_dir)(d, scale=SCALE) for d in tqdm(dirs))
        for i in range(len(imgdata)):
            imgdata[i] = imgdata[i]

        imgdata = np.stack(imgdata)
        np.save(f_out, imgdata)
    else:
        print("Using previously scaled frames ...")
        imgdata = np.load(f_out)

    ids = [op.basename(d).split('_')[0] for d in dirs]
    
    pca = PCA(n_components=N_COMPS)
    colnames = [f'comp_{str(i+1).zfill(3)}' for i in range(N_COMPS)]
    
    ### Save PCA-reduced set
    for norm in [False, True]:
        if norm:  # normalize pixels!
            # Normalize across all samples (dim 0) and frames (dim 1)
            nz = ~np.isclose(imgdata.std(axis=(0, 1)), 0.)
            imgdata[:, :, nz] -= imgdata[:, :, nz].mean(axis=(0, 1))
            imgdata[:, :, nz] /= imgdata[:, :, nz].std(axis=(0, 1))

        ns = 'y' if norm else 'n'

        pca.fit(imgdata[:, 0, :])
        np.savez(f'models/featurespace-pixelPCA_frame-01_norm-{ns}_weights.npz', mu=pca.mean_, w=pca.components_)
        df_f01 = pd.DataFrame(pca.transform(imgdata[:, 0, :]), index=index, columns=colnames)
        df_f01['data_split'] = info.loc[df_f01.index, 'data_split']
        df_f01.to_csv(f'data/featurespace-pixelPCA_frame-01_norm-{ns}.tsv', sep='\t')

        pca.fit(imgdata[:, 14, :])
        np.savez(f'models/featurespace-pixelPCA_frame-15_norm-{ns}_weights.npz', mu=pca.mean_, w=pca.components_)
        df_f15 = pd.DataFrame(pca.transform(imgdata[:, 14, :]), index=index, columns=colnames)
        df_f15['data_split'] = info.loc[df_f15.index, 'data_split']
        df_f15.to_csv(f'data/featurespace-pixelPCA_frame-15_norm-{ns}.tsv', sep='\t')

        pca.fit(imgdata.reshape(len(dirs) * 30, imgdata.shape[2]))
        np.savez(f'models/featurespace-pixelPCA_frame-all_norm-{ns}_weights.npz', mu=pca.mean_, w=pca.components_)

        change = pca.transform(imgdata[:, 14, :]) - pca.transform(imgdata[:, 0, :])
        df_f15min01 = pd.DataFrame(change, columns=colnames, index=index)
        df_f15min01['data_split'] = info.loc[df_f15min01.index, 'data_split']
        df_f15min01.to_csv(f'data/featurespace-pixelPCA_frame-15min01_norm-{ns}.tsv', sep='\t')

        diff = imgdata[:, 14, :] - imgdata[:, 0, :]
        pca.fit(diff)
        np.savez(f'models/featurespace-pixelPCA_frame-diff_norm-{ns}_weights.npz', mu=pca.mean_, w=pca.components_)
        df_f15min01 = pd.DataFrame(pca.transform(diff), columns=colnames, index=index)
        df_f15min01['data_split'] = info.loc[df_f15min01.index, 'data_split']
        df_f15min01.to_csv(f'data/featurespace-pixelPCA_frame-diff_norm-{ns}.tsv', sep='\t')