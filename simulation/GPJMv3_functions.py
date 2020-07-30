# GPJM v3: Spatiotemporal GPJM
# Auxilliary functions

import numpy as np
import gpflow
import tensorflow as tf

def recover_latent(m, ts_new):
    import tensorflow as tf
    from numpy.linalg import inv, cholesky
    ts = tf.Session().run(m.ts)
    Kstar = m.kern_tX.compute_K(ts, ts_new)
    KttI = m.kern_tX.compute_K(ts, ts) + np.eye(ts.shape[0], dtype=np.float64) * m.likelihood_tX.variance.read_value()
    X = m.X.read_value()
    L = cholesky(KttI)
    return Kstar.T.dot(inv(L.T).dot((inv(L)).dot(X)))

def recover_Kxn(m, ts_new):
    X_new = recover_latent(m, ts_new)
    X = m.X.read_value()
    ss = tf.Session().run(m.ss)
    kern_temporal = m.kern_XN.kernel_t.compute_K(X, X_new)
    kern_spatial = m.kern_XN.kernel_s.compute_K(ss, ss)
    ts_N = tf.Session().run(m.ts_N)
        
    i, k, s = ss.shape[0], ts_N.shape[0], ts_N.shape[0]
    o = s * (i - 1) + k
    with tf.Session() as sess:
        Kss = tf.reshape(kern_spatial, [1, i, i, 1])
        Ktt = tf.reshape(kern_temporal, [k, k, 1, 1])
        Kst = tf.squeeze(tf.nn.conv2d_transpose(Kss, Ktt, (1, o, o, 1), [1, s, s, 1], "VALID"))
        res = sess.run(Kst)
    return res

def recover_neural(m, ts_new):
    import tensorflow as tf
    from numpy.linalg import inv, cholesky
    ts = tf.Session().run(m.ts)
    ts_N = tf.Session().run(m.ts_N)
    ss = tf.Session().run(m.ss)
    Y_N = tf.Session().run(m.Y_N)
    Kstar = recover_Kxn(m, ts_new)
    KttI = recover_Kxn(m, ts) + np.eye(Y_N.shape[0], dtype = np.float64) * m.likelihood_XN.variance.read_value()
    L = cholesky(KttI)
    fmean = Kstar.T.dot(inv(L.T).dot((inv(L)).dot(Y_N)))
    v = inv(L).dot(Kstar)
    Vstar = Kstar - v.T.dot(v)
    sd = np.sqrt(np.diag(Vstar))
    return fmean, fmean.ravel().reshape(ss.shape[0], ts_N.shape[0]).T, Vstar, sd.ravel().reshape(ss.shape[0], ts_N.shape[0]).T

def recover_behavioral(m, ts_new):
    import tensorflow as tf
    from numpy.linalg import inv, cholesky
    X_new = recover_latent(m, ts_new)
    X = m.X.read_value()
    Kstar = m.kern_XB.compute_K(X, X_new)
    KttI = m.kern_XB.compute_K(X, X) + np.eye(X.shape[0], dtype = np.float64) * m.likelihood_XB.variance.read_value()
    Y_B = tf.Session().run(m.Y_B)
    L = cholesky(KttI)
    fmean = Kstar.T.dot(inv(L.T).dot((inv(L)).dot(Y_B)))
    
    v = inv(L).dot(Kstar)
    Vstar = m.kern_XB.compute_K(X_new, X_new) - v.T.dot(v)
    ci95 = np.column_stack([fmean - 1.96 * np.sqrt(np.diag(Vstar)).reshape(-1,1), fmean + 1.96 * np.sqrt(np.diag(Vstar)).reshape(-1,1)])
    return fmean, Vstar, ci95