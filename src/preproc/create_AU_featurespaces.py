import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('../stims/stimuli-expressive_selection-all.csv', sep='\t', index_col=0)
df = df.sort_index()

au_cols = sorted([col for col in df.columns if col[:2] == 'AU'])
df = df.loc[:, au_cols] #+ ['data_split']]
df = df.drop('AU22', axis=1)  # issue with animation!
df.to_csv('data/featurespace-AU.tsv', sep='\t', index=True)

# AU pooled
mapping = {
    'AU1': 'eyebrows',
    'AU2': 'eyebrows',
    'AU4': 'eyebrows',
    'AU5': 'eyes',
    'AU6': 'eyes',
    'AU7': 'eyes',
    'AU9': 'nose',
    'AU10Open': 'mouth',
    'AU11': 'nose',
    'AU12': 'mouth',
    'AU13': 'mouth',
    'AU14': 'mouth',
    'AU15': 'mouth',
    'AU16Open': 'mouth',
    'AU17': 'mouth',
    'AU20': 'mouth',
    'AU24': 'mouth',
    'AU25': 'mouth',
    'AU26': 'mouth',
    'AU27i': 'mouth'
}

poly = PolynomialFeatures(degree=2, interaction_only=True , include_bias=False)
X_AUxAU = poly.fit_transform(df.iloc[:, :-1].values)
feature_names = poly.get_feature_names(df.iloc[:, :-1].columns)
X_AUxAU = pd.DataFrame(X_AUxAU, columns=feature_names, index=df.index)
#X_AUxAU.loc[:, 'data_split'] = df.copy()['data_split']
X_AUxAU.to_csv('data/featurespace-AUxAU.tsv', sep='\t', index=True)