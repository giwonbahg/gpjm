# GPJM v3: Spatiotemporal GPJM
# Data generation code

import numpy as np
import gpflow
import tensorflow as tf

def kernel_mSE(theta, x, xstar):
    sigma_f = theta[0]
    l = theta[1]
    K = np.zeros((x.shape[0], xstar.shape[0]))
    for i in range(x.shape[0]):
        for j in range(xstar.shape[0]):
            K[i,j] = sigma_f**2 * np.exp(-0.5*(x[i,] - xstar[j,]).T.dot((x[i,] - xstar[j,]))/(l))
    return K

def kernel_mSE_ARD(theta, x, xstar):
    sigma_f = theta[0]
    l = theta[1:(1+x.shape[1])]
    K = np.zeros((x.shape[0], xstar.shape[0]))
    
    for i in range(x.shape[0]):
        for j in range(xstar.shape[0]):
            temp_dist = np.zeros((xstar.shape[1]))
            for k in range(xstar.shape[1]):
                temp_dist[k] = (x[i,k] - xstar[j,k])**2 / (l[k])
            K[i,j] = sigma_f**2 * np.exp(-0.5 * np.sum(temp_dist))
    return K


def kernel_mMatern12(theta, x, xstar):
    sigma_f = theta[0]
    l = theta[1]
    K = np.zeros((x.shape[0], xstar.shape[0]))
    for i in range(x.shape[0]):
        for j in range(xstar.shape[0]):
            K[i,j] = sigma_f**2 * np.exp(-np.sqrt(np.sum((x[i,] - xstar[j,])**2))/l)
    return K

def kernel_mMatern12_ARD(theta, x, xstar):
    sigma_f = theta[0]
    l = theta[1:(1+xstar.shape[1])]
    K = np.zeros((x.shape[0], xstar.shape[0]))
    for i in range(x.shape[0]):
        for j in range(xstar.shape[0]):
            temp_dist = np.zeros((xstar.shape[1]))
            for k in range(xstar.shape[1]):
                temp_dist[k] = np.sqrt(np.sum((x[i,k] - xstar[j,k])**2))/l[k]
            K[i,j] = sigma_f**2 * np.exp(-np.sum(temp_dist))
    return K

def oneD_conv(V, kernel):
    len_k = len(kernel)
    len_V = len(V)
    zeros = np.zeros((len_k-1))
    V2 = np.concatenate([zeros, V, zeros])
    res = []
    for i in range(len_V):
        res.append(V2[i:(i+len_k)].dot(kernel))
    return(np.array(res))

def row_conv(K, k):
    K2 = np.zeros_like(K)
    for i in range(K.shape[0]):
        K2[i,:] = oneD_conv(K[i,:], k)
    return(K2)

def row_col_conv(K,k):
    return row_conv(row_conv(K,k).T, k).T

def data_gen(ts, ss, theta_cc, theta_ss, theta_bb, sigma2, n_samp, seed = 1):
    sigma2_n = sigma2[0]
    sigma2_b = sigma2[1]
    n_n = n_samp[0]
    n_b = n_samp[1]
    # Latent dynamics kernel
    #cs = np.column_stack([np.squeeze(np.sin(ts)), 2 * np.squeeze((np.sin(ts.T/2.5))**58 + 0.05 * np.sin(ts.T))])
    cs = np.column_stack([np.squeeze(np.sin(ts/8)), np.squeeze(np.sin(ts/4))])
    Kcc = kernel_mSE_ARD(theta_cc, cs, cs)
    Kcc_b = kernel_mMatern12_ARD(theta_bb, cs, cs)
    def define_HRF_filter_temp():
        unit_ts = np.arange(0, 30, 2, dtype = np.float64)
        def HRFunit(t):
            from scipy.special import gamma
            a1 = 6 # b1=1
            a2 = 16 # b2=1
            c = 1./6
            part1 = t**(a1-1) * np.exp(-t) / gamma(a1)
            part2 = t**(a2-1) * np.exp(-t) / gamma(a2)
            return part1 - c * part2
        hrf = HRFunit(unit_ts)
        return(hrf)
    hrf = np.flip(define_HRF_filter_temp())
    Kcc_conv = row_col_conv(Kcc, hrf)
    # Spatial kernel
    Kss = kernel_mSE(theta_ss, ss, ss)
    Ksc = np.kron(Kss, Kcc_conv)
    # Kronecker kernel
    KIn = Ksc + sigma2_n * np.diag(np.ones(Ksc.shape[0]))
    
    # Generate the data
    np.random.seed(seed)
    yn = np.random.multivariate_normal(np.zeros(Ksc.shape[0]), KIn, n_n).reshape((ss.shape[0], ts.shape[0])).T
    KIb = Kcc_b + sigma2_b * np.diag(np.ones(Kcc.shape[0]))
    np.random.seed(seed)
    yb = np.random.multivariate_normal(np.zeros(Kcc.shape[0]), KIb, n_b)
    yb = yb - yb.mean()
    return yn, KIn, yb, KIb, cs, Kcc, Kcc_conv, Kss

def downsample_neuraldata(yn, ts_N, ts_dense):
    M = np.zeros((ts_dense.shape[0], ts_N.shape[0]))
    ts = np.squeeze(ts_dense)
    for i in range(ts_N.shape[0]):
        argmin_idx = np.argmin(np.abs(ts - ts_N[i,0]))
        M[argmin_idx,i] = 1
    m_idx = M.sum(axis=1).astype(bool)
    yn_new = yn[m_idx,:]
    return yn_new, m_idx, M