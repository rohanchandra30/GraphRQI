import numpy as np
import scipy.sparse.linalg as sla
import time
from sklearn.decomposition import TruncatedSVD


def incremental_svd(U, S, V, a, b, force_orth=False):
    start_time = time.time()
    current_rank = U.shape[1]
    m = np.matmul(U.transpose(), a)
    p = a - np.matmul(U, m)
    Ra = np.sqrt(np.matmul(p.transpose(), p))
    P = (1 / Ra) * p

    # if Ra < 1e-13:
    #     print('------> Whoa! No orthogonal component of m!')

    n = np.matmul(V.transpose(), b)
    q = b - np.matmul(V, n)
    Rb = np.sqrt(np.matmul(q.transpose(), q))
    Q = (1 / Rb) * q

    # if Rb < 1e-13:
    #     print('------> Whoa! No orthogonal component of n!')

    z = np.zeros(m.shape)
    K = np.append(np.append(np.diag(S), z, axis=1), np.expand_dims(np.append(z.transpose(), 0), axis=0), axis=0) +\
        np.matmul(np.append(m, Ra, axis=0), np.append(n, Rb, axis=0).transpose())
    tUp, tSp, tVp = sla.svds(K, current_rank)
    tVp = tVp.transpose()
    Sp = tSp
    Up = np.matmul(np.append(U, P, axis=1), tUp)
    Vp = np.matmul(np.append(V, Q, axis=1), tVp)

    if force_orth:
        UQ, UR = np.linalg.qr(Up, mode='economic')
        VQ, VR = np.linalg.qr(Vp, mode='economic')
        tUp, tSp, tVp = sla.svds(np.matmul(UR, np.matmul(Sp, VR.transpose())), current_rank)
        tVp = tVp.transpose()
        Up = np.matmul(UQ, tUp)
        Vp = np.matmul(VQ, tVp)
        Sp = tSp
    return Up, Sp, Vp, time.time() - start_time


def restart_svd(old_U, old_S, old_V, del_A):
    start_time = time.time()
    N, K = old_U.shape
    old_X = old_U
    for i in range(K):
        temp_i = np.argmax(np.abs(old_X[:, i]))
        if old_X[temp_i, i] < 0:
            old_X[:, i] = - old_X[:, i]
    temp_v = np.amax(old_U, axis=0)
    temp_i = np.argmax(old_U, axis=0)
    old_V_unraveled = old_V.transpose().reshape(-1)
    temp_sign = np.sign(temp_v * old_V_unraveled[np.ravel_multi_index((temp_i, np.arange(K)), (N, K), order='F')])
    old_L = old_S * temp_sign

    temp_sum = np.matmul(old_X.transpose(), np.matmul(del_A, old_X))
    del_L = np.diag(temp_sum)
    del_X = np.zeros((N, K))
    for i in range(K):
        temp_D = np.diag(old_L[i] + del_L[i] - old_L)
        temp_alpha = np.matmul(np.linalg.pinv(temp_D - temp_sum), temp_sum[:, i])
        del_X[:, i] = np.matmul(old_X, temp_alpha)

    new_U = old_X + del_X
    for i in range(K):
        new_U[:, i] = new_U[:, i] / np.sqrt(np.matmul(new_U[:, i].transpose(), new_U[:, i]))
    new_S = np.abs(old_L + del_L)
    new_V = np.matmul(new_U, np.diag(np.sign(old_L + del_L)))
    return new_U, new_S, new_V, time.time() - start_time


def truncated_svd(X, n_components):
    start_time = time.time()
    svd = TruncatedSVD(n_components=min(X.shape[0] - 1, n_components))
    svd.fit(X)
    return time.time() - start_time


def cg(A, b, x=None):
    """
    Description
    -----------
    Solve a linear equation Ax = b with conjugate gradient method.
    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    b: 1d numpy.array
    x: 1d numpy.array of initial point
    Returns
    -------
    1d numpy.array x such that Ax = b
    """
    n = len(b)
    if not x:
        x = np.ones([n,1])
    r = np.dot(A, x) - b
    p = - r
    # r_k_norm = np.dot(r, r)
    r_k_norm = np.linalg.norm ( r )*np.linalg.norm ( r )
    for i in range(2*n):
        Ap = np.dot(A, p)
        alpha = r_k_norm / p.T@Ap
        try:
            x += alpha * p
        except:
            pass
        r += alpha * Ap
        r_kplus1_norm = np.linalg.norm ( r )*np.linalg.norm ( r )
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < 1e-5:
            break
        p = beta * p - r
    return x

def eigenvalue(A, eigvec):
    return np.dot(np.transpose(eigvec), A.dot(eigvec))/np.dot(np.transpose(eigvec), eigvec)


def inverse_power_method(A):
    start_time = time.time()
    for _ in range(A.shape[0]):
        eigenvec = np.random.rand(A.shape[0], 1)
        eigenval = eigenvalue(A, eigenvec)
        for _ in range(1):
            new_eigenvec,_ = sla.cg(A, eigenvec)
            new_eigenvec = np.expand_dims(new_eigenvec,axis=1)
            new_eigenvec /= np.linalg.norm(new_eigenvec)
            new_eigenval = eigenvalue(A, new_eigenvec)
            if np.linalg.norm(new_eigenval-eigenval) < np.finfo(float).eps:
                return new_eigenvec
            eigenvec = new_eigenvec
            eigenval = new_eigenval
    return time.time() - start_time