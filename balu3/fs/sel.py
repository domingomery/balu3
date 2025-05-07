import numpy as np
from itertools import combinations
import warnings
import tqdm
import math

def jfisher(features, ypred, p=None): 
    m = features.shape[1]
    
    norm = ypred.ravel() - ypred.min()
    max_class = norm.max() + 1
    
    if p is None:
        p = np.ones(shape=(max_class, 1)) / max_class
        
    # Centroid of all samples
    features_mean = features.mean(0)

    # covariance within class 
    cov_w = np.zeros(shape=(m, m))
    
    # covariance between classes
    cov_b = np.zeros(shape=(m, m))

    for k in range(max_class):
        ii = (norm == k)                                   # indices from class k
        class_features = features[ii,:]                    # samples of class k
        class_mean = class_features.mean(0)                # centroid of class k 
        class_cov = np.cov(class_features, rowvar=False)   # covariance of class k
        
        cov_w += p[k] * class_cov                          # within-class covariance
        
        dif = (class_mean - features_mean).reshape((m, 1))
        cov_b += p[k] * dif @ dif.T                        # between-class covariance
    try:
        return np.trace(np.linalg.inv(cov_w) @ cov_b)
    except np.linalg.LinAlgError:
        return - np.inf


def sp100(features, ypred):
    norm = (ypred.flatten() -  ypred.min()).astype(bool)
    max_class = norm.max()

    if max_class > 1:
        raise Exception('sp100 works only for two classes')

    c1_features = features[norm == 1,:]

    min_feat = c1_features.min(0)
    max_feat = c1_features.max(0)

    z1 = (features >= min_feat) & (features <= max_feat)
    dr = z1.all(1)
    
    TP = (dr * norm).sum()
    FP = dr.sum() - TP

    TN = (~dr * ~norm).sum()
    return TN / (FP+TN)


def score(features, ypred, *, method='fisher', param=None):
    if param is None:
        dn = ypred.max() - ypred.min() + 1  # number of classes
        p = np.ones((dn, 1)) / dn
    else:
        p = param

    if method == 'mi':  # mutual information
        raise NotImplementedError()

    # maximal relevance
    elif method == 'mr':
        raise NotImplementedError()

    # minimal redundancy and maximal relevance
    elif method == 'mrmr':
        raise NotImplementedError()

    # fisher
    elif method == 'fisher':
        return jfisher(features, ypred, p)

    elif method == 'sp100':
        return sp100(features, ypred)

    else:
        return 0

def clean(features, show=False):
    n_features = features.shape[1]
    ip = np.ones(n_features, dtype=int)

    # cleaning correlated features
    warnings.filterwarnings('ignore')
    C = np.abs(np.corrcoef(features, rowvar=False))
    idxs = np.vstack(np.where(C > .99))
    
    # remove pairs of same feature ( feature i will have a correlation of 1 whit itself )
    idxs = idxs[:, idxs[0,:] != idxs[1,:]]
    
    # remove correlated features
    if idxs.size > 0:
        ip[np.max(idxs, 0)] = 0
    
    # remove constant features
    s = features.std(axis=0, ddof=1)
    ip[s < 1e-8] = 0
    p = np.where(ip.astype(bool))[0]

    if show:
        print(f'Clean: number of features reduced from {n_features} to {p.size}.')

    return p


def sfs(features, ypred, n_features, *, force=False, method='fisher', options=None, show=False):

    N, M = features.shape
    remaining_feats = set(np.arange(M))
    selected = list()
    curr_feats = np.zeros((N, 0))
    if options is None:
        options = dict()

    def _calc_score(i):
        feats = np.hstack([curr_feats, features[:, i].reshape(-1, 1)])
        return score(feats, ypred, method=method, **options)

    if show:
        _range = tqdm.trange(
            n_features, desc='Selecting Features', unit_scale=True, unit=' features')
    else:
        _range = range(n_features)

    for _ in _range:
        new_selected = max(remaining_feats, key=_calc_score)
        selected.append(new_selected)
        remaining_feats.remove(new_selected)
        curr_feats = np.hstack(
            [curr_feats, features[:, new_selected].reshape(-1, 1)])

    return np.array(selected)

#def choose(n, k):
#    return int(np.math.factorial(n) / (np.math.factorial(n - k) * np.math.factorial(k)))
#
#def exsearch(features, ypred, n_features, *, method='fisher', options=None, show=False):
#    if options is None:
#        options = dict()
#
#    tot_feats = features.shape[1]
#    N = choose(tot_feats, n_features)
#
#    if N > 10000:
#        warnings.warn(
#            f'Doing more than 10.000 iterations ({N}). This may take a while...')
#
#    def _calc_score(ii):
#        feats = features[:, ii]
#        return score(feats, ypred, method=method, **options)
#
#    _combinations = combinations(range(tot_feats), n_features)
#
#    if show:
#        _combinations = zip(tqdm.trange(N,
#                                        desc='Combinations checked',
#                                        unit_scale=True,
#                                        unit=' combinations'),
#                            _combinations)
#
#        _combinations = (ii for _, ii in _combinations)
#
#    chosen_feats = max(_combinations, key=_calc_score)
#
#    return np.array(chosen_feats)





def choose(n, k):
    """Calculate the binomial coefficient."""
    # Use scipy.special.comb for numerical stability with large numbers
    # import scipy.special
    # return int(scipy.special.comb(n, k))
    # Or use the formula directly with numpy for factorial
    return int(math.factorial(n) / (math.factorial(n - k) * math.factorial(k)))


def exsearch(features, ypred, n_features, *, method='fisher', options=None, show=False):
    if options is None:
        options = dict()

    tot_feats = features.shape[1]
    N = choose(tot_feats, n_features)

    if N > 10000:
        warnings.warn(
            f'Doing more than 10.000 iterations ({N}). This may take a while...')

    def _calc_score(ii):
        feats = features[:, ii]
        # Use jfisher to calculate the score
        return jfisher(feats, ypred)  
    _combinations = combinations(range(tot_feats), n_features)

    if show:
        # Assuming tqdm is imported somewhere else in the user's notebook
        #_combinations = zip(tqdm.trange(N,
        #                                desc='Combinations checked',
        #                                unit_scale=True,
        #                                unit=' combinations'),
        #                    _combinations)

        _combinations = (ii for _, ii in _combinations)

    chosen_feats = max(_combinations, key=_calc_score)

    return np.array(chosen_feats)