import sys
import logging
import os.path as op
import pandas as pd
import numpy as np
from tqdm import tqdm 
from copy import copy
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.append('src')
from noise_ceiling import convert_doubles_to_single_labels
from utils import argmax_with_random_ties


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(funcName)-8.8s] [%(levelname)-7.7s]  %(message)s",
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()
    ]
)


DU2EN = dict(
    Verrassing='surprise',
    Verdrietig='sadness',
    Bang='fear',
    Walging='disgust',
    Blij='happiness',
    Boos='anger'
)


class DataLoader:
    
    def __init__(self, sub='01', data_dir=None, rnd_seed=42, log_level=20):
        """ Initializes a DataLoader object.

        Parameters
        ----------
        sub : str
            Subject ID (zero-padded)
        data_dir : str
            Path to data directory
        rnd_seed : int
            Random seed for train-test set split
        """
        self.sub = sub
        self.y = None
        self.X = None
        self.target_name = None 

        if data_dir is None:
            data_dir = op.abspath('data')
            if not op.isdir(data_dir):
                raise ValueError(f"Directory {data_dir} does not exist.")

        self.data_dir = data_dir
        self.rnd_seed = rnd_seed
        self.le = LabelEncoder().fit(['happiness', 'surprise', 'fear', 'sadness', 'disgust', 'anger'])
        self.ohe = OneHotEncoder(
            categories=self.le.classes_,
            sparse=False
        )
        self.log = logging.getLogger(__name__)
        self.log.setLevel(log_level)

    def load_y(self, target='emotion', data_split='train', filter_gva=True, strategy_doubles='soft', return_doubles=False):
        """ Loads the target variable (y). 
        
        Parameters
        ----------
        target : str
            Name of target variable ("emotion", "valence", or "arousal")
        data_split : str
            Either "train" or "test"
        filter_gva : bool
            Whether to remove the "geen van allen" ratings
        strategy_doubles : str
            Strategy to handle doubles (either 'soft', 'hard', or 'none').
        """

        self.target_name = target

        f = op.join(self.data_dir, 'ratings_complete', f'sub-{self.sub}_task-expressive_ratings.tsv')
        df = pd.read_csv(f, sep='\t', index_col=0)
        df = df.query("rating_type == @target")  # filter rating type

        n_orig = df.shape[0]
        df = df.query("data_split == @data_split")
        self.log.info(f"Removed {n_orig - df.shape[0]} test trials (N = {df.shape[0]}).")
        
        if filter_gva:  # remove "geen van allen" (none of all)
            n_orig = df.shape[0]
            df = df.query("rating != 'Geen van allen'")
            n_remov = n_orig - df.shape[0]
            self.log.info(f"Removed {n_remov} 'Geen van allen' trials (N = {df.shape[0]}).")

        with pd.option_context('mode.chained_assignment', None):  # suppress stupid warning
            df.loc[:, 'rating'] = df.loc[:, 'rating'].replace(DU2EN)  # translate
            df.loc[:, 'rating'] = self.le.transform(df['rating'])

        # Handle doubles for session 1
        if return_doubles:
            dup_idx = df.index[df.index.duplicated()].unique()
            dups = df.loc[dup_idx, :].sort_index()
            return dups['rating']

        if strategy_doubles in ['hard', 'soft']:  # stupid slow code
            soft = True if strategy_doubles == 'soft' else False
            df = convert_doubles_to_single_labels(df['rating'], soft=soft, keepdims=False)

            if strategy_doubles == 'hard':
                # Isn't this redundant, because I can do convert_doubles ... with soft=False?
                df = pd.DataFrame(df.values.argmax(axis=1), columns=['rating'], index=df.index)

            """
            n_orig = df.shape[0]
            df = df.drop(dups.index, axis=0)  # remove duplicates

            filt_dups = []
            for i, idx in enumerate(dup_idx):
                tmp = dups.loc[idx, :]
                counts = tmp['rating'].value_counts()
                if strategy_doubles == 'argmax':
                    maxcount = counts.loc[counts == counts.max()].sample(n=1).index[0]
                    filt_dups.append(tmp.loc[tmp['rating'] == maxcount, :].sample(n=1))
            
            filt_dups = pd.concat(filt_dups, axis=0)
            df = pd.concat((df, filt_dups), axis=0)
            """
            self.log.info(f"Removed {n_orig - df.shape[0]} duplicates (N = {df.shape[0]}).")

        if strategy_doubles == 'soft':
            # WARNING: source of randomness
            df_tmp = pd.DataFrame(argmax_with_random_ties(df.values), columns=['rating'], index=df.index)
            
            df_tmp_train, df_test = train_test_split(df, test_size=0.15, random_state=self.rnd_seed, stratify=df_tmp['rating'])
            df = df.loc[df_tmp_train.index, :]
            self.y = df['rating']
            #self.intensity = df['intensity']
        else:
            df, df_test = train_test_split(df, test_size=0.15, random_state=self.rnd_seed, stratify=df['rating'])
            self.y = df['rating']
            #self.intensity = df['intensity']

        self.log.info(f"Split into train (N = {df.shape[0]}) and test (N = {df_test.shape[0]}).")
        self.rating_df = df

    def load_X(self, feature_set, n_comp=50):
        """ Loads in predictors/independent variables (X).

        Parameters
        ----------
        feature_set : str/list/tuple
            Name of one or more feature-sets to use as predictors
        n_comp : int
            Number of components 
        """

        if not isinstance(feature_set, (list, tuple)):
            feature_set = (feature_set,)

        if self.y is None:
            raise ValueError("Call load_y before load_X!")

        X = []
        for fs in feature_set:  # load in 1 or more feature-sets
            self.log.info(f"Loading feature-set {fs}.")
            path = op.join(self.data_dir, f'featurespace-{fs}.tsv')
            df = pd.read_csv(path, sep='\t', index_col=0)
            
            if 'sub' in df.columns:
                df = df.query(f"sub == 'sub-{self.sub}'").drop('sub', axis=1)

            """
            if 'rep' in df.columns:
                tmp_index = self.rating_df.index + '_' + self.rating_df['rep'].astype(str)
                df.index = df.index + '_' + df['rep'].astype(str)
            else:
                tmp_index = self.rating_df.index
            """
            #df = df.loc[tmp_index, :].set_index(self.rating_df.index)
            df = df.loc[self.rating_df.index, :].set_index(self.rating_df.index)

            if 'data_split' in df.columns:
                df = df.drop('data_split', axis=1)

            if 'pca' in fs or 'nmf' in fs:
                df = df.iloc[:, :n_comp]

            X.append(df)

        X = pd.concat(X, axis=1)

        self.X = X
        self.log.info(f"Shape X: {self.X.shape}.")

    def return_Xy(self):
        """ Returns the labels (y) and predictors (X). """
        return self.X, self.y
