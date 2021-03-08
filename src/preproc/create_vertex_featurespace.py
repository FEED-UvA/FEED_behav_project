import os.path as op
import numpy as np
import pandas as pd
from glob import glob
from scipy.io import loadmat
from tqdm import tqdm
from sklearn.decomposition import PCA
from joblib import Parallel, delayed


N_COMPS = 50
info = pd.read_csv('../stims/stimuli-expressive_selection-all.csv', sep='\t', index_col=0)

dirs = sorted(glob('../FEED_stimulus_frames/id*'))
ids = np.array([op.basename(d).split('_')[0].split('-')[1] for d in dirs], dtype=str)
N = len(dirs)
index = [op.basename(d) for d in dirs]
ids = [f.split('_')[0] for f in index]
dm_ids = pd.get_dummies(ids).to_numpy()
#dm_ids = dm_ids.repeat(30).reshape((848 * 30, 50))

def load_vertices(d):
    files = sorted(glob(f'{d}/*/texmap/*vertices.mat'))
    vertices = np.zeros((30, 31049, 3))
    for ii, f in enumerate(files):
        vertices[ii, :] = loadmat(f)['vertices']
    
    return vertices

if op.isfile('data/vertices.npy'):
    print("Loading vertices")
    vertices = np.load('data/vertices.npy')
    print('done')
else:
    out = Parallel(n_jobs=30)(delayed(load_vertices)(d) for d in tqdm(dirs))
    vertices = np.stack(out)
    np.save('data/vertices.npy', vertices)

v_stats = dict()
v_stats['mean'] = vertices[:, 0, :, :].mean(axis=0)
v_stats['std'] = vertices[:, 0, :, :].std(axis=0)
v_stats['mean_face'] = np.zeros((31049, 3, 50))
for i in range(3):
    v_stats['mean_face'][:, i, :] = np.linalg.lstsq(dm_ids, vertices[:, 0, :, i], rcond=None)[0].T

print(v_stats['mean_face'].shape)
np.savez('data/vertices_stats.npz', **v_stats)
exit()
#v_stats['mean_face'] = np.zeros((50, 31049 * 3))
#for i in tqdm(range(tmp.shape[-1])):
#    v_stats['mean_face'][:, i] = np.linalg.lstsq(dm_ids, tmp[:, i], rcond=None)[0]
#v_stats['mean_face'] = v_stats['mean_face'].reshape((50, 31049, 3))

#tmp = vertices.reshape((848 * 30, 31049 * 3))
#b = np.linalg.lstsq(dm_ids, tmp, rcond=None)[0]
#tmp -= dm_ids @ b

pca = PCA(n_components=N_COMPS)

### RAW FEATURES
N = vertices.shape[0]
# tmp = vertices[:, 0, :, :].reshape((N, 31049*3))
# colnames = [f'v{i}' for i in range(tmp.shape[1])]
# df = pd.DataFrame(tmp, columns=colnames, index=index)
# df.to_csv('data/featurespace-vertices_frame-01.tsv', sep='\t')

# tmp = vertices[:, 14, :, :].reshape((N, 31049*3))
# df = pd.DataFrame(tmp, columns=colnames, index=index)
# df.to_csv('data/featurespace-vertices_frame-15.tsv', sep='\t')

# tmp = vertices[:, 14, :, :].reshape((N, 31049*3)) - vertices[:, 0, :, :].reshape((N, 31049*3))
# df = pd.DataFrame(tmp, columns=colnames, index=index)
# df.to_csv('data/featurespace-vertices_frame-15min01.tsv', sep='\t')

# overall_motion = np.sqrt(np.sum(np.diff(vertices, axis=1) ** 2, axis=(1, 2, 3)))
# df = pd.DataFrame(overall_motion, columns=['overall_motion'], index=index)
# df.to_csv('data/featurespace-vertexoverallmotion.tsv', sep='\t')

# # Summed motion: N x vertices
# motion = np.sqrt((np.diff(vertices, axis=1) ** 2).sum(axis=3)).sum(axis=1)
# df = pd.DataFrame(motion, columns=[f'motion{i}' for i in range(motion.shape[1])], index=index)
# df.to_csv('data/featurespace-vertexmotion.tsv', sep='\t')

#### PCA STUFF ####
colnames = [f'comp_{str(i).zfill(3)}' for i in range(N_COMPS)]
pca = PCA(n_components=N_COMPS)

### Frame 01
for norm in [False, True]:

    if norm:
        print("Normalizing vertices for PCA ...")
        # Note to self: because of the different face IDs,
        # all the vertices actually have a standard deviation != 0
        vertices -= vertices.mean(axis=(0, 1))
        vertices /= vertices.std(axis=(0, 1))

    ns = 'y' if norm else 'n'
    colnames = [f'comp_{str(i).zfill(3)}' for i in range(N_COMPS)]

    print("Fitting vertex-based frame 01 ...")
    pca.fit(vertices[:, 0, :, :].reshape((N, 31049 * 3)))
    np.savez(f'models/featurespace-vertexPCA_frame-01_norm-{ns}_weights.npz', mu=pca.mean_, w=pca.components_)
    df_f01 = pd.DataFrame(pca.transform(vertices[:, 0, :, :].reshape((N, 31049 * 3))), columns=colnames, index=index)
    #df_f01['data_split'] = info.loc[df_f01.index, 'data_split']
    df_f01.to_csv(f'data/featurespace-vertexPCA_frame-01_norm-{ns}.tsv', sep='\t')

    # Frame 15
    print("Fitting vertex-based frame 15 ...")
    pca.fit(vertices[:, 14, :, :].reshape((N, 31049 * 3)))
    np.savez(f'models/featurespace-vertexPCA_frame-15_norm-{ns}_weights.npz', mu=pca.mean_, w=pca.components_)
    df_f15 = pd.DataFrame(pca.transform(vertices[:, 14, :, :].reshape((N, 31049 * 3))), columns=colnames, index=index)
    #df_f15['data_split'] = info.loc[df_f15.index, 'data_split']
    df_f15.to_csv(f'data/featurespace-vertexPCA_frame-15_norm-{ns}.tsv', sep='\t')

    # diff
    print("Fitting vertex-based all frames ...")
    pca.fit(vertices.reshape((N * 30, 31049 * 3)))
    np.savez(f'models/featurespace-vertexPCA_frame-all_norm-{ns}_weights.npz', mu=pca.mean_, w=pca.components_)
    change = pca.transform(vertices[:, 14, :, :].reshape((N, 31049 * 3))) - pca.transform(vertices[:, 0, :].reshape((N, 31049 * 3)))
    df_f15min01 = pd.DataFrame(change, columns=colnames, index=index)
    #df_f15min01['data_split'] = info.loc[df_f15min01.index, 'data_split']
    df_f15min01.to_csv(f'data/featurespace-vertexPCA_frame-15min01_norm-{ns}.tsv', sep='\t')

    # diff
    print("Fitting vertex-based frame 15 min 01 ...")
    change = vertices[:, 14, :, :] - vertices[:, 0, :, :]
    pca.fit(change.reshape((N, 31049 * 3)))
    np.savez(f'models/featurespace-vertexPCA_frame-diff_norm-{ns}_weights.npz', mu=pca.mean_, w=pca.components_)
    df_f15min01 = pd.DataFrame(pca.transform(change.reshape((N, 31049 * 3))), columns=colnames, index=index)
    #df_f15min01['data_split'] = info.loc[df_f15min01.index, 'data_split']
    df_f15min01.to_csv(f'data/featurespace-vertexPCA_frame-diff_norm-{ns}.tsv', sep='\t')

    # ### MOTION-RELATED FEATURES ###
    # # motion over time (frames, axis=1)
    motion = np.sqrt((np.diff(vertices, axis=1) ** 2).sum(axis=3)).sum(axis=1)
    print("Fitting vertex-based motion ...")
    
    pca.fit(motion)
    np.savez(f'models/featurespace-vertexmotionPCA_norm-{ns}_weights.npz', mu=pca.mean_, w=pca.components_)

    df = pd.DataFrame(pca.transform(motion), columns=colnames, index=index)
    df.to_csv(f'data/featurespace-vertexmotionPCA_norm-{ns}.tsv', sep='\t')