import copy
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Lasso, Ridge

def get_flatten_cube(factor_value_cube):
    K = factor_value_cube.shape[2]
    factor_value_matrix = []
    for k in range(K):
        factor_value = factor_value_cube[:,:,k]
        if np.nansum(~np.isnan(factor_value)) == 0: continue
        factor_value_matrix.append(truncnormalize(factor_value).flatten())
    return np.array(factor_value_matrix).T

def get_cube(factor_value_matrix, shape):
    factor_value_cube = []
    for i in range(factor_value_matrix.shape[1]):
        factor_value_cube.append(factor_value_matrix[:,i].reshape(shape))
    return np.dstack(factor_value_cube)

def orthogonalize(factor_value_cube, method = 'symmetric'):
    shape = (factor_value_cube.shape[0], factor_value_cube.shape[1])
    F = get_flatten_cube(factor_value_cube)
    F = (F - np.nanmean(F, axis = 0)) / np.nanstd(F, axis = 0)
    F[np.isnan(F)] = 0
    M = F.T @ F 
    d, U = np.linalg.eig(M) 
    D = np.diag(d)
    if method == 'PCA':
        F_hat = PCA().fit_transform(F)
        return get_cube(F_hat, shape)
    elif method == 'symmetric':
        S = U @ np.linalg.inv(sqrtm(D)) @ U.T
    elif method == 'canonical':
        S = U @ np.linalg.inv(sqrtm(D))
    F_hat = F @ S
    return get_cube(F_hat, shape)

def pca_combine(factor_value_cube):
    F = get_flatten_cube(factor_value_cube)
    F = (F - np.nanmean(F, axis = 0)) / np.nanstd(F, axis = 0)
    F[np.isnan(F)] = 0
    return PCA().fit_transform(F)[:,0].reshape((factor_value_cube.shape[0],factor_value_cube.shape[1]))
    # mask = np.nansum(np.isnan(F),axis=1) == 0
    # input = F[mask]
    # output = np.full(F.shape[0],np.nan)
    # output[mask] = PCA().fit_transform(input)[:,0]
    # return output.reshape((factor_value_cube.shape[0],factor_value_cube.shape[1]))

def equal_weighted_combine(factor_value_cube):
    factor_value_list = []
    for i in range(factor_value_cube.shape[2]):
        factor_value = factor_value_cube[:,:,i]
        if np.nansum(~np.isnan(factor_value)) == 0: continue
        factor_value_list.append(truncnormalize(factor_value))
    return np.nanmean(np.dstack(factor_value_list), axis = 2)

def ic_weighted_combine(factor_value_cube, rtn):
    K = factor_value_cube.shape[2]
    factor_value_list = []
    ic_list = []
    for k in range(K):
        factor_value = factor_value_cube[:,:,k]
        if np.nansum(~np.isnan(factor_value)) == 0: continue
        transform = truncnormalize(factor_value)
        factor_value_list.append(transform)
        ic_list.append(np.nanmean(corr_by_row(transform,rtn)))
    weights = np.array(ic_list) / np.nansum(ic_list)
    weighted_factor_value_list = []
    for i in range(len(weights)):
        weighted_factor_value_list.append(weights[i] * factor_value_list[i])
    return np.nansum(np.dstack(weighted_factor_value_list),axis=2)

def ic_ir_weighted_combine(factor_value_cube, rtn):
    K = factor_value_cube.shape[2]
    factor_value_list = []
    ic_ir_list = []
    for k in range(K):
        factor_value = factor_value_cube[:,:,k]
        if np.nansum(~np.isnan(factor_value)) == 0: continue
        transform = truncnormalize(factor_value)
        factor_value_list.append(transform)
        ic_ir_list.append(np.nanmean(corr_by_row(transform,rtn))/np.nanstd(corr_by_row(transform,rtn)))
    weights = np.array(ic_ir_list) / np.nansum(ic_ir_list)
    weighted_factor_value_list = []
    for i in range(len(weights)):
        weighted_factor_value_list.append(weights[i] * factor_value_list[i])
    return np.nansum(np.dstack(weighted_factor_value_list),axis=2)

def corr_by_row(m1, m2):
    m1 = copy.deepcopy(m1); m2 = copy.deepcopy(m2)
    if type(m1)==list: m1 = np.array(m1)
    if type(m2)==list: m2 = np.array(m2)
    mask = (np.isnan(m1)) | (np.isnan(m2))
    m1[mask] = np.nan; m2[mask] = np.nan
    d1 = m1 - np.nanmean(m1,axis=1)[:,np.newaxis]
    d2 = m2 - np.nanmean(m2,axis=1)[:,np.newaxis]
    s1 = np.sqrt(np.nansum(d1*d1,axis=1))
    s2 = np.sqrt(np.nansum(d2*d2,axis=1))
    cov = np.nansum(d1*d2,axis=1)
    s1[s1==0] = np.nan
    s2[s2==0] = np.nan
    return cov / s1 / s2
    
def max_ic_ir_weighted_combine(factor_value_cube,rtn):
    T = factor_value_cube.shape[0]
    K = factor_value_cube.shape[2]
    ic_tss = np.full((T,K),np.nan)
    for k in range(K):
        factor_value = factor_value_cube[:,:,k]
        if np.nansum(~np.isnan(factor_value)) == 0: continue
        ic_tss[:,k] = corr_by_row(factor_value,rtn)
    mu = np.nanmean(ic_tss, axis = 0)
    Sigma = pd.DataFrame(ic_tss).cov().values
    w = np.linalg.inv(Sigma) @ mu; w[w<0] = 0
    return w / np.nansum(w)

def max_ic_weighted_combine(factor_value_cube,rtn):
    T = factor_value_cube.shape[0]
    N = factor_value_cube.shape[1]
    K = factor_value_cube.shape[2]
    ic_tss = np.full((T,K),np.nan)
    fac_val = np.full((N,K),np.nan)
    for k in range(K):
        factor_value = factor_value_cube[:,:,k]
        if np.nansum(~np.isnan(factor_value)) == 0: continue
        ic_tss[:,k] = corr_by_row(factor_value,rtn)
        fac_val[:,k] = factor_value[-1,:]
    mu = np.nanmean(ic_tss, axis = 0)
    v = pd.DataFrame(fac_val).cov().values
    w = np.linalg.inv(v) @ mu; w[w<0] = 0
    return w / np.nansum(w)

def linear_regression(factor_value_cube,rtn,method,alpha=1):
    N = factor_value_cube.shape[1]
    X = get_flatten_cube(factor_value_cube)[:-N,:]
    y = truncnormalize(rtn).flatten()[N:]
    mask = (np.nansum(np.isnan(X),axis=1) > 0) | np.isnan(y)
    if mask.all(): return np.full(N,np.nan)
    X = X[~mask]; y = y[~mask]
    if method == 'ols':
        model = LinearRegression()
    elif method == 'lasso':
        model = Lasso(alpha=alpha)
    elif method == 'ridge':
        model = Ridge(alpha=alpha)
    model.fit(X,y)
    X_os = get_flatten_cube(factor_value_cube)[-N:,:]
    y_os = np.full(N,np.nan)
    mask_os = np.nansum(np.isnan(X_os),axis=1) > 0
    X_os = X_os[~mask_os]
    y_os[~mask_os] = model.predict(X_os)
    return y_os

def cal_tot_rtn(fac_val,rtn):
    transform = truncnormalize(fac_val)
    transform = transform / np.nansum(transform * (transform > 0), axis = 1).reshape((-1,1))
    mask = np.nansum(~np.isnan(transform),axis=1) == 0 # all nan
    tot_rtn = np.nansum(pd.DataFrame(transform).shift(1).values * rtn, axis = 1)
    tot_rtn[mask] = np.nan
    return tot_rtn

def cal_stats(rtn,ds):
    ix = np.where(~np.isnan(rtn))[0][0]
    prd_rtn = np.nansum(rtn[ix:])
    ann_rtn = np.nanmean(rtn[ix:]) * 25
    ann_vol = np.nanstd(rtn[ix:]) * np.sqrt(25) 
    shrp = ann_rtn / ann_vol
    nav = np.cumsum(rtn[ix:])
    mdd, mdd_bgn, mdd_end = 0, 0, 0
    for i in range(1,len(nav)):
        dd_i = np.full(i,nav[i]) - nav[:i]
        mdd_i = np.nanmin(dd_i)
        if mdd_i <= mdd:
            mdd = mdd_i
            mdd_bgn = np.argmin(dd_i)
            mdd_end = i
    wrt = np.nansum(rtn[ix:] > 0) / len(rtn[ix:])
    return {'prd_rtn': prd_rtn * 100,
            'ann_rtn': ann_rtn * 100,
            'ann_vol': ann_vol * 100,
            'shrp': shrp,
            'mdd': mdd * 100,
            'mdd_bgn': ds[ix:][mdd_bgn],
            'mdd_end': ds[ix:][mdd_end],
            'wrt': wrt * 100}
