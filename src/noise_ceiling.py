import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from tqdm import tqdm


def convert_doubles_to_single_labels(y, K=None, soft=True, keepdims=False, doubles_only=False):
    """ Convert labels from repeated trials to a single label.

    Parameters
    ----------
    y : pd.Series
        Series with numeric values and index corresponding to stim ID.
    K : int
        Number of classes (if categorical; does not work for regression yet)
    soft : bool
        Whether to use a probabilistic estimate (True) or "hard" label (False)
    keepdims : bool
        Whether to reduce the doubles to singles (False) or to repeat the
        best prediction across repetitions (True)
 
    Returns
    -------
    labels : DataFrame
        DataFrame of shape (samples x classes)
    """

    if K is None:
        K = y.unique().size

    # Get unique indices (stim IDs)
    #uniq_idx = y.index[y.index.duplicated()].unique()
    uniq_idx = y.index.unique()
    counts = np.zeros((uniq_idx.size, K))
    for i, idx in enumerate(uniq_idx):
        if isinstance(y.loc[idx], pd.core.series.Series):
            counts_df = y.loc[idx].value_counts()
            counts[i, counts_df.index] = counts_df.values
        else:
            counts[i, y.loc[idx]] = 1

    if soft:
        labels = counts / counts.sum(axis=1, keepdims=True)
    else:   
        # Pre-allocate best prediction array
        labels = np.zeros_like(counts, dtype=float)

        for ii in range(counts.shape[0]):
            # Determine most frequent label across reps
            opt_class = np.where(counts[ii, :] == counts[ii, :].max())[0]
            rnd_class = np.random.choice(opt_class, size=1)  
            labels[ii, rnd_class] = 1

    # Repeat best possible prediction R times
    labels = pd.DataFrame(labels, index=uniq_idx)
    if doubles_only:
        doubles_idx = counts.sum(axis=1) != 1
        labels = labels.loc[uniq_idx[doubles_idx], :]

    if keepdims:
        labels = labels.loc[y.index, :]

    return labels


def compute_noise_ceiling(y, scoring=roc_auc_score, K=None, soft=True, return_pred=False, doubles_only=False):
    """ Computes the noise ceiling for data with repetitions.

    Parameters
    ----------
    y : pd.Series
        Series with numeric values and index corresponding to stim ID.
    K : int
        Number of classes (if categorical; does not work for regression yet)
    soft : bool
        Whether to use a probabilistic estimate (True) or "hard" label (False)
    
    Returns
    -------
    ceiling : ndarray
        Numpy ndarray (shape: K,) with ceiling estimates
    """
    
    labels = convert_doubles_to_single_labels(y, K, soft, keepdims=True, doubles_only=doubles_only)    
    labels = labels.sort_index()
    y_flat = y.loc[labels.index.unique()].sort_index()

    # Needed to convert 1D to 2D
    ohe = OneHotEncoder(categories='auto', sparse=False)
    y_ohe = ohe.fit_transform(y_flat.values[:, np.newaxis])

    # Compute best possible score ("ceiling")
    ceiling = scoring(y_ohe, labels.values, average=None)

    if return_pred:
        return ceiling, labels
    else:
        return ceiling


class SimulationDataset:
    
    def __init__(self, P, N_per_class, K, R, verbose=False):
        self.P = P
        self.N_per_class = N_per_class
        self.N = N_per_class * K
        self.K = K
        self.R = R
        self.verbose = verbose
        self.rep_idx = np.repeat(np.arange(self.N), R)
        self.y = None
        self.X = None
        self.ohe = None  # added later
    
    def generate(self, signal=0.1, inconsistency=5):
        """ Generates pseudo-random data (X, y).
        
        Parameters
        ----------
        signal : float
            "Amount" of signal added to X to induce corr(X, y)
        """

        index = np.arange(self.N)
        X_unrep = pd.DataFrame(np.random.normal(0, 1, (self.N, self.P)), index=index)
        self.X = pd.concat([X_unrep for _ in range(self.R)])  # repeat R times!
        
        # Generate "unrepeated" labels
        y_unrep = np.repeat(np.arange(self.K), self.N_per_class)
        
        # Generate random labels, repeated R times, simulating
        # inconsistency in subject ratings (sometimes anger, sometimes happiness, etc.)
        shifts = np.random.normal(0, inconsistency, R).astype(int)
        self.y = pd.concat([pd.Series(np.roll(y_unrep, shifts[ii]), index=index) for ii in range(R)])

        # Add signal
        for k in range(self.K):
            tmp_idx = (self.y == k).values.squeeze()
            self.X.loc[tmp_idx, k] = self.X.loc[tmp_idx, k] + signal
            
        self.ohe = OneHotEncoder(sparse=False, categories='auto')
        self.ohe.fit(y_unrep[:, np.newaxis])
        
    def compute_noise_ceiling(self, soft=True):
        """ Estimates the best prediction and score (noise ceiling) given the 
        inconsistency in the labels.

        Parameters
        ----------
        use_prob : bool
            Whether to evaluate probabilistic performance or binarized
        """

        self.ceiling = compute_noise_ceiling(
            self.y,
            soft=soft
        )
        
        if self.verbose:
            print(f"Ceiling: {np.round(self.ceiling, 2)}")

    def compute_model_performance(self, estimator, use_prob=True, cv=None, stratify_reps=False):
        
        # Fit actual model
        if cv is None:
            estimator.fit(self.X, self.y)
            preds = estimator.predict(self.X)
        else:
            preds = cross_val_predict(
                estimator, self.X, self.y, cv=cv,
                groups=self.rep_idx if stratify_reps else None
            )
        
        # Compute actual score (should be > ceiling)
        y_ohe = self.ohe.transform(self.y[:, np.newaxis])
        self.score = roc_auc_score(
            self.ohe.transform(self.y[:, np.newaxis]),
            self.ohe.transform(preds[:, np.newaxis]),
            average=None if use_prob else 'micro'
        )
        
        if self.verbose:
            print(f"Score: {np.round(self.score, 2)}")
        
        self.diff = self.ceiling - self.score


if __name__ == '__main__':
    P = 1000  # number of features
    N_per_class = 10  # number of samples per class
    K = 3  # number of classes [0 - K]
    R = 4  # how many repetitions of each sample
    estimator = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    iters = 100

    ds = SimulationDataset(P=P, N_per_class=N_per_class, K=K, R=R, verbose=False)

    scores = np.zeros((iters, K))
    ceilings = np.zeros((iters, K))
    diffs = np.zeros((iters, K))

    for i in tqdm(range(iters)):
        ds.generate(signal=0, inconsistency=5)
        ds.compute_noise_ceiling()
        ds.compute_model_performance(estimator)