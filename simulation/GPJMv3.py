# GPJM v2: Spatially independent GPJM
# The model code for cross-validation

import numpy as np
import gpflow
import tensorflow as tf

# Generate a convolved RBF kernel 
class KernelHRFConvDownsized_RBF(gpflow.kernels.Kernel):
    def __init__(self, input_dim, ts_N, ts_dense):
        super().__init__(input_dim = input_dim)
        # Set the basis kernel
        self.kernel = gpflow.kernels.RBF(input_dim = input_dim, ARD = True)
        # Set the HRF
        def HRF_filter(ts_dense):
            ts = np.squeeze(ts_dense)
            unit_ts = ts[ts <= 30]
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
        self.hrf_dense = tf.constant(np.flip(HRF_filter(ts_dense)))
        self.len_hrf_dense = self.hrf_dense.shape[0]
        # Set the downsizing scheme
        def downsizing_scheme_nearest(ts_N, ts_dense):
            M = np.zeros((ts_dense.shape[0], ts_N.shape[0]))
            ts = np.squeeze(ts_dense)
            for i in range(ts_N.shape[0]):
                argmin_idx = np.argmin(np.abs(ts - ts_N[i,0]))
                M[argmin_idx,i] = 1
            return M
        self.M = tf.constant(downsizing_scheme_nearest(ts_N, ts_dense))

    def K(self, X, X2=None, presliced=False):
        if X2 is None:
            X2=X
        padK = tf.pad(self.kernel.K(X, X2), [[self.len_hrf_dense-1, 0],[self.len_hrf_dense-1,0]], 'CONSTANT')
        hrf4conv = tf.reshape(self.hrf_dense, [self.len_hrf_dense, 1, 1])
        temp0 = tf.reshape(padK, [tf.shape(padK)[0], tf.shape(padK)[1], 1], name='temp0')
        res0 = tf.squeeze(tf.nn.conv1d(temp0, hrf4conv, 1, 'VALID'))
        temp1 = tf.reshape(tf.transpose(res0), [tf.shape(res0)[1], tf.shape(res0)[0], 1], name='temp1')
        res1 = tf.transpose(tf.squeeze(tf.nn.conv1d(temp1, hrf4conv, 1, 'VALID')))
        # Downsample the dense kernel
        res2 = tf.matmul(tf.transpose(self.M), tf.matmul(res1, self.M))
        return(res2)

# A spatiotemporal kernel for 
class KernelKronecker_Neural(gpflow.kernels.Kernel):
    def __init__(self, input_dim_n, ts_N, ts_B, ss, kernel_temporal = KernelHRFConvDownsized_RBF, kernel_spatial = gpflow.kernels.RBF):
        super().__init__(input_dim = input_dim_n + ss.shape[1])
        self.kernel_t = kernel_temporal(input_dim = input_dim_n, ts_N = ts_N, ts_dense = ts_B)
        self.kernel_s = kernel_spatial(input_dim = ss.shape[1])
        self.ts_N = ts_N
        self.ts_B = ts_B
        self.ss = ss
        
    def K(self, X, X2=None, presliced=False):
        Xt = X[0]
        Xs = X[1]
        Xt2 = X2[0]
        Xs2 = X2[1]
        kern_temporal = self.kernel_t.K(Xt, Xt2)
        kern_spatial = self.kernel_s.K(Xs, Xs2)
        # Kronecker product        
        i, k, s = self.ss.shape[0], self.ts_N.shape[0], self.ts_N.shape[0]
        o = s * (i - 1) + k
        Kss  = tf.reshape(kern_spatial, [1, i, i, 1])
        Ktt = tf.reshape(kern_temporal, [k, k, 1, 1])
        Kst = tf.squeeze(tf.nn.conv2d_transpose(Kss, Ktt, (1, o, o, 1), [1, s, s, 1], "VALID"))
        return(Kst)

class GPJMv3(gpflow.models.Model):
    def __init__(self, Y_N, Y_B, ts_N, ts_B, n_latent, ss, neural_kernel = KernelKronecker_Neural, conv_scheme = KernelHRFConvDownsized_RBF,
                 kern_tX = None, mean_tX = None, kern_XN = None, mean_XN = None, kern_XB = None, mean_XB = None, name=None):
        if kern_tX is None:
            kern_tX = gpflow.kernels.RBF(input_dim=1)
        if mean_tX is None:
            mean_tX = gpflow.mean_functions.Zero(output_dim = n_latent)
        if kern_XN is None:
            kern_XN = neural_kernel(input_dim_n = n_latent, ts_N = ts_N, ts_B = ts_B, ss = ss, kernel_temporal = conv_scheme)
        if mean_XN is None:
            mean_XN = gpflow.mean_functions.Zero(output_dim = Y_N.shape[1])
        if kern_XB is None:
#             kern_XB = gpflow.kernels.RBF(input_dim = n_latent)
            kern_XB = gpflow.kernels.Matern12(input_dim = n_latent, ARD=True)
        if mean_XB is None:
            mean_XB = gpflow.mean_functions.Zero(output_dim = Y_B.shape[1])
        super().__init__(name=name)
        
        def cubic_interpolation(ts_sparse, Y_N, ts_dense, ss):
            from scipy import interpolate
            yn_new = np.zeros((ts_dense.shape[0], ss.shape[0]))
            yn_array = Y_N.reshape(ss.shape[0], ts_sparse.shape[0]).T
            for i in range(ss.shape[0]):
                temp = interpolate.interp1d(np.squeeze(ts_sparse), yn_array[:,i], kind='cubic')
                yn_new[:,i] = temp(np.squeeze(ts_dense))
            return yn_new
        
        def downsizing_scheme_nearest(ts_sparse, ts_dense):
            M = np.zeros((ts_dense.shape[0], ts_sparse.shape[0]))
            ts = np.squeeze(ts_dense)
            for i in range(ts_sparse.shape[0]):
                argmin_idx = np.argmin(np.abs(ts - ts_sparse[i,0]))
                M[argmin_idx,i] = 1
            return M
        
        def HRF_filter(ts_dense):
            ts = np.squeeze(ts_dense)
            unit_ts = ts[ts <= 30]
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
        
        if len(ts_N) > len(ts_B):
            print("Neural: Dense / Behavioral: Sparse")
            self.ts = tf.constant(ts_N.copy())
            self.ts_np = ts_N.copy()
        elif len(ts_N) < len(ts_B):
            print("Neural: Sparse / Behavioral: Dense")
            self.ts = tf.constant(ts_B.copy())
            self.ts_np = ts_B.copy()
            self.Y_N_interp = cubic_interpolation(ts_N, Y_N, ts_B, ss)
            self.M = downsizing_scheme_nearest(ts_N, ts_B)
        
        # Data
        self.Y_N = tf.constant(Y_N.copy())
        self.Y_B = tf.constant(Y_B.copy())
        self.ts_N = tf.constant(ts_N.copy())
        self.ts_B = tf.constant(ts_B.copy())
        self.ss = tf.constant(ss.copy())
        self.n_Nsample = Y_N.shape[0]
        self.n_Nfeature = Y_N.shape[1]
        self.n_Bsample = Y_B.shape[0]
        self.n_Bfeature = Y_B.shape[1]
        
        # latent dynamics kernel + downsizing scheme
        self.kern_tX = kern_tX
        self.mean_tX = mean_tX
        self.n_latent = n_latent
        self.N_pca = gpflow.models.gplvm.PCA_reduce(self.Y_N_interp, n_latent)
        self.X = gpflow.Param(gpflow.models.gplvm.PCA_reduce(self.Y_N_interp, n_latent))

        # Neural data kernel
        self.kern_XN = kern_XN
        self.mean_XN = mean_XN
        self.hrf = tf.constant(HRF_filter(self.ts_np))
        
        # Behavioral data kernel
        self.kern_XB = kern_XB
        self.mean_XB = mean_XB

        # Likelihood
        self.likelihood_tX = gpflow.likelihoods.Gaussian()
        self.likelihood_XN = gpflow.likelihoods.Gaussian()
        self.likelihood_XB = gpflow.likelihoods.Gaussian() # Can differ according to the model you rely on.
    
    @gpflow.params_as_tensors
    def _build_likelihood_tX(self): # Zero-noise model is not supported by GPflow ==> Need to add an infinitesimal noise when initializing the model.
        Ktx = self.kern_tX.K(self.ts, self.ts) + tf.eye(tf.shape(self.ts)[0], dtype = gpflow.settings.float_type) * self.likelihood_tX.variance
        Ltx = tf.cholesky(Ktx)
        mtx = self.mean_tX(self.ts)
        logpdf_tx = gpflow.logdensities.multivariate_normal(self.X, mtx, Ltx)
        return tf.reduce_sum(logpdf_tx)
    
    @gpflow.params_as_tensors
    def _build_likelihood_XN(self):
        Kxn = self.kern_XN.K([self.X, self.ss], [self.X, self.ss]) + tf.eye(self.n_Nsample, dtype = gpflow.settings.float_type) * self.likelihood_XN.variance
        Lxn = tf.cholesky(Kxn)
        mxn = self.mean_XN(self.Y_N)
        logpdf_xn = gpflow.logdensities.multivariate_normal(self.Y_N, mxn, Lxn)
        return tf.reduce_sum(logpdf_xn)

    @gpflow.params_as_tensors
    def _build_likelihood_XB(self):
        Kxb = self.kern_XB.K(self.X, self.X) + tf.eye(tf.shape(self.X)[0], dtype = gpflow.settings.float_type) * self.likelihood_XB.variance
        Lxb = tf.cholesky(Kxb)
        mxb = self.mean_XB(self.X)
        logpdf_xb = gpflow.logdensities.multivariate_normal(self.Y_B, mxb, Lxb)
        return tf.reduce_sum(logpdf_xb)

    @gpflow.name_scope('likelihood')
    @gpflow.params_as_tensors
    def _build_likelihood(self):
        logpdf_tx = self._build_likelihood_tX()
        logpdf_xn = self._build_likelihood_XN()
        logpdf_xb = self._build_likelihood_XB()
        return tf.reduce_sum(logpdf_tx + logpdf_xn + logpdf_xb)