import pandas as pd

info = pd.read_csv('../stims/stimuli-expressive_selection-all.csv', sep='\t', index_col=0)
info = info.loc[:, info.columns.str.contains('face_1')]
info.to_csv('data/featurespace-faceID.tsv', sep='\t')

au = pd.read_csv('data/featurespace-AU.tsv', sep='\t', index_col=0)
id_au = pd.concat((info, au.loc[info.index, :]), axis=1)
id_au.to_csv('data/featurespace-faceID+AU.tsv', sep='\t')
