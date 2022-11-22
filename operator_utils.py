import numpy as np
import pandas as pd
import bottleneck as bn
from scipy.stats import pearsonr, spearmanr
from progressbar import ProgressBar as pb
from joblib import Parallel, delayed
import scipy.cluster.hierarchy as sch

def fold(factor,med=True):
    if med == True:
        diff = factor - np.nanmedian(factor,axis=1).reshape((-1,1))
    else:
        diff = factor - np.nanmean(factor,axis=1).reshape((-1,1))
    return diff * np.sign(diff)

def tscorr(x, y, w, m):
    if m == 0:
        mu_x = bn.move_mean(x, w, int(w/4), axis = 0)
        mu_y = bn.move_mean(y, w, int(w/4), axis = 0)
        mu_xy = bn.move_mean(x*y, w, int(w/4), axis = 0)
        sigma_x = bn.move_std(x, w, int(w/4), axis = 0)
        sigma_y = bn.move_std(y, w, int(w/4), axis = 0)
        return (mu_xy - mu_x * mu_y) / (sigma_x * sigma_y)
    elif m == 1:
        mu = np.nanmean(range(w)+1)
        std = np.nanstd(range(w)+1)
        mu_pv = np.zeros(x.shape); mu_pv[:] = np.nan
        res = Parallel(n_jobs=10)(delayed(calc)(i,x[i-w:i,:],y[i-w:i,:]) for i in pb()(range(w,x.shape[0])))
        for i, pv in res:
            mu_pv[i-1,:] = pv
        corr = (mu_pv - mu ** 2) / (std ** 2)
        return corr

def calc(i,p,v):
    p_rank = bn.rankdata(p,axis=0)
    v_rank = bn.rankdata(v,axis=0)
    return i, np.nanmean(p_rank*v_rank,axis=0)

def rank(array):
    return pd.DataFrame(array).rank(axis = 1, method = 'dense', pct = True).values - 0.5

def truncnormalize(array, thres = 0.01):
    fac_val = array.copy()
    left = np.nanquantile(fac_val,thres,axis=1).reshape((-1,1)); left_mask = fac_val < left
    right = np.nanquantile(fac_val,1-thres,axis=1).reshape((-1,1)); right_mask = fac_val > right
    fac_val[left_mask] = np.nan; fac_val[right_mask] = np.nan
    return (fac_val - np.nanmean(fac_val,axis=1).reshape((-1,1))) / np.nanstd(fac_val,axis=1).reshape((-1,1))

def winsorize(array, p = 5, type = 2):
    upper = np.nanquantile(array, (100 - p) / 100, axis = 1).reshape((-1,1))
    lower = np.nanquantile(array, p / 100, axis = 1).reshape((-1,1))
    if type == 2:
        array[(array >= upper)|(array <= lower)] = np.nan
    elif type == 1:
        array[array >= upper] = np.nan
    elif type == -1:
        array[array <= lower] = np.nan
    return array

def normalize(array):
    mu = np.nanmean(array,axis=1).reshape((-1,1))
    sigma = np.nanstd(array,axis=1).reshape((-1,1))
    score = (array - mu) / sigma
    return score.clip(-3,3)

def corr_table(cube):
    T = cube.shape[0]
    corr_ls = []
    for t in range(T):
        cur = cube[t,:,:]
        corr_ls.append(pd.DataFrame(cur).corr().values)
    return np.nanmean(np.dstack(corr_ls),axis=2)

def corr_table_flatten(cube, norm=True):
    N = cube.shape[0] * cube.shape[1]
    K = cube.shape[2]
    mat = np.zeros((N,K)); mat[:] = np.nan
    for k in range(K):
        if norm == True:
            mat[:,k] = truncnormalize(cube[:,:,k]).flatten()
        else:
            mat[:,k] = cube[:,:,k].flatten()
    return pd.DataFrame(mat).corr().values

def cluster_corr(corr_table, inplace=False):
    pairwise_distances = sch.distance.pdist(corr_table)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max()/2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold, criterion='distance')
    idx = np.argsort(idx_to_cluster_array)
    if not inplace:
        corr_table = corr_table.copy()
    if isinstance(corr_table, pd.DataFrame):
        return corr_table.iloc[idx, :].T.iloc[idx, :]
    return corr_table[idx, :][:, idx]
