import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.special import softmax
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

"""
IMotions=dict(
    happiness=['AU6', 'AU12'],
    sadness=['AU1', 'AU4', 'AU15'],
    surprise=['AU1', 'AU2', 'AU5', 'AU26'],
    fear=['AU1', 'AU2', 'AU4', 'AU5', 'AU7', 'AU20', 'AU26'],
    anger=['AU4', 'AU5', 'AU7'],  # misses AU23
    disgust=['AU9', 'AU15', 'AU16Open']
),
FaceReader=dict(
    happiness=['AU6', 'AU12'],
    sadness=['AU1', 'AU4'],
    surprise=['AU1', 'AU2'],
    fear=['AU1', 'AU2', 'AU4'],
    anger=['AU4', 'AU5'],
    disgust=['AU10Open', 'AU25']
),
"""
theory_kernels = dict(

    Darwin=dict(
        happiness=['AU6', 'AU12'],
        sadness=['AU1', 'AU15'],
        surprise={
            1: ['AU1', 'AU2', 'AU5', 'AU25'],
            2: ['AU1', 'AU2', 'AU5', 'AU26']
        },
        fear=['AU1', 'AU2', 'AU5', 'AU20'],
        anger=['AU4', 'AU5', 'AU24'],  # misses AU38
        disgust={
            1: ['AU10Open', 'AU16Open', 'AU22', 'AU25'],
            2: ['AU10Open', 'AU16Open', 'AU22', 'AU26']
        }
    ),
    Matsumoto2008=dict(
        happiness=['AU6', 'AU12'],
        sadness={
            1: ['AU1', 'AU15'],
            2: ['AU4'],
            3: ['AU4', 'AU1', 'AU15'],
            4: ['AU17'],
            5: ['AU17', 'AU1', 'AU15'],
            6: ['AU17', 'AU1', 'AU15', 'AU4']
        },
        surprise={
            1: ['AU1', 'AU2', 'AU5', 'AU25'],
            2: ['AU1', 'AU2', 'AU5', 'AU26']
        },
        fear={
            1: ['AU1', 'AU2', 'AU4', 'AU5', 'AU20'],
            2: ['AU25'],
            3: ['AU26']
        },
        anger={
            1: ['AU4', 'AU5', 'AU22', 'AU24'],  # misses AU23
            2: ['AU4', 'AU7', 'AU22', 'AU24']   # misses AU23
        },
        disgust={
            1: ['AU9'],
            2: ['AU10Open'],
            3: ['AU25'],
            4: ['AU26'],
            5: ['AU9', 'AU25'],
            6: ['AU10Open', 'AU26']
        }
    ),
    Keltner2019=dict(
        happiness=['AU6', 'AU7', 'AU12', 'AU25', 'AU26'],
        sadness=['AU1', 'AU4', 'AU6', 'AU15', 'AU17'],
        surprise=['AU1', 'AU2', 'AU5', 'AU25', 'AU26'],
        fear=['AU1', 'AU2', 'AU4', 'AU5', 'AU7', 'AU20', 'AU25'],
        anger=['AU4', 'AU5', 'AU17', 'AU24', ],  # misses AU23 
        disgust=['AU7', 'AU9', 'AU25', 'AU26']  # misses AU19, tongue show
    ),
    Cordaro2008ref=dict(
        happiness=['AU6', 'AU12'],
        sadness=['AU1', 'AU4', 'AU5'],
        surprise=['AU1', 'AU2', 'AU5', 'AU26'],
        fear=['AU1', 'AU2', 'AU4', 'AU5', 'AU7', 'AU20', 'AU26'],
        anger=['AU4', 'AU5', 'AU7'],  # misses AU23
        disgust=['AU9', 'AU15', 'AU16Open']
    ),
    Cordaro2008IPC=dict(
        happiness={
            1: ['AU6', 'AU7', 'AU12', 'AU16Open', 'AU25', 'AU26'],
            2: ['AU6', 'AU7', 'AU12', 'AU16Open', 'AU25', 'AU27i'],
        },
        sadness=['AU4', 'AU43'],  # misses AU54 (head down)
        surprise=['AU1', 'AU2', 'AU5', 'AU25'],
        fear=['AU1', 'AU2', 'AU5', 'AU7', 'AU25'],  # also "jaw"/"move back"
        anger=['AU4', 'AU7'],
        disgust=['AU4', 'AU6', 'AU7', 'AU9', 'AU10Open', 'AU25', 'AU26']  # also "jaw"
    )
)


def _softmax_2d(arr, beta):

    scaled = beta * arr
    num = np.exp(scaled - scaled.max(axis=1, keepdims=True))
    denom = np.sum(num, axis=1, keepdims=True)
    return num / denom


class TheoryKernelClassifier(BaseEstimator, ClassifierMixin):  
    """ A "Theory-kernel classifier" that computes the probability of an emotion
    being present in the face based on the associated theoretical action unit
    configuration. """
    def __init__(self, au_dict, binarize_X=False, normalize=True, scale_dist=True, beta_sm=3):
        """ Initializes an TheoryKernelClassifier object.
        
        au_dict : dict
            Dictionary with theoretical AU configuration
        binarize_X : bool
            Whether to binarize the configuration (0 > = 1, else 0)
        normalize : bool
            Whether to normalize the distances with the sum of the AU-vector
        scale_dist : bool
            Whether to scale the distances from 0 to 1
        beta_sm : int/float
            Beta parameter for the softmax function
        """

        self.normalize = normalize
        self.binarize_X = binarize_X
        self.scale_dist = scale_dist
        self.beta_sm = beta_sm
        self.aus = ['AU1', 'AU10Open', 'AU11', 'AU12', 'AU13', 'AU14', 'AU15',
                    'AU16Open', 'AU17', 'AU2', 'AU20', 'AU22', 'AU24', 'AU25',
                    'AU26', 'AU27i', 'AU4', 'AU43', 'AU5', 'AU6', 'AU7', 'AU9']

        self.au_dict = au_dict
        for emo, cfg in self.au_dict.items():
            if isinstance(cfg, dict):#
                vec = np.zeros((len(cfg), len(self.aus)))
                for i, combi in cfg.items():
                    for c in combi:
                        vec[i-1, self.aus.index(c)] = 1
            else:
                vec = np.zeros(len(self.aus))
                for c in cfg:
                    vec[self.aus.index(c)] = 1

            setattr(self, emo, vec)
            
        le = LabelEncoder().fit(['happiness', 'surprise', 'fear', 'sadness', 'disgust', 'anger'])
        self.le = le
        self.ohe = OneHotEncoder(categories='auto', sparse=False)
        self.ohe.fit(le.transform(le.classes_)[:, np.newaxis])
        self.classes_ = self.le.classes_
        
    def fit(self, X, y=None):
        pass
    
    def predict_proba(self, X, y=None):
        return self._predict(X, y)

    def predict(self, X, y=None, string=False):
        probs = self._predict(X, y)
        ties = np.squeeze(probs == probs.max())
        if np.sum(ties) > 0:
            pred = np.random.choice(np.arange(ties.size)[ties])
        else:
            pred = np.argmax(probs)
            
        if string:
            pred = self.le.inverse_transform([pred])

        return pred
        
    def _predict(self, X, y=None):
        
        if X.ndim != 2:
            raise ValueError("X is not 2D.")
        
        if X.shape[1] != 22:
            raise ValueError("This classifier only works with exactly 22 AUs.")
        
        if self.binarize_X:
            X = (X > 0).astype(int)
            
        self.distances = np.zeros((X.shape[0], 6))
        for key in self.au_dict.keys():
            idx = self.le.transform([key])[0]
            vec = getattr(self, key)
            
            if vec.ndim > 1:
                dist_all = np.zeros((vec.shape[0], X.shape[0]))
                for i in range(vec.shape[0]):
                    sqdist = (X - vec[i, :]) ** 2
                    if self.normalize:
                        sqdist = sqdist / vec[i, :].sum()
                    
                    dist_all[i, :] = np.sqrt(np.sum(sqdist, axis=1))
                dist = np.max(dist_all,  axis=0)  # or mean() ???
            else:
                sqdist = (X - vec) ** 2
                if self.normalize:
                    sqdist = sqdist / vec.sum()
                
                dist = np.sqrt(np.sum(sqdist, axis=1))
            
            self.distances[:, idx] = dist
        
        if self.scale_dist:
            rnge = self.distances.max(axis=1) - self.distances.min(axis=1)
            self.distances = (self.distances - self.distances.min(axis=1, keepdims=True)) / rnge[:, np.newaxis]
        
        EPS = 1e-10
        self.distances[self.distances == 0] = EPS
        #sm_dist = softmax(1 / self.distances, axis=1)
        #print((1 / self.distances)[:5, :])
        sm_dist = _softmax_2d(1 / self.distances, self.beta_sm)
        return sm_dist
