import numpy as np
from svd_methods import incremental_svd, restart_svd, truncated_svd, inverse_power_method, eigenvalue, cg
from scipy import linalg as LA
import time

min_size = 3
max_size = 100
X = np.zeros((max_size, max_size))
# X[:min_size, :min_size] = np.random.rand(min_size, min_size)
X[:max_size, :max_size] = np.random.rand(max_size, max_size)
time_i = 0
time_r = 0
time_t = 0
time_svd = 0
time_cg = 0
sizes = range(min_size, max_size+1)
time_i_mean = 0
time_r_mean = 0
time_t_mean = 0
time_svd_mean = 0
time_cg_mean = 0
trials = range(5)
for t in trials:
    for s in sizes:
        a = np.random.rand(max_size, 1)
        b = np.random.rand(max_size, 1)
        del_A = np.matmul(a, b.transpose())
        time_svd_start = time.time()
        U, S, V = LA.svd(X, lapack_driver='gesvd')
        time_svd +=  time.time() - time_svd_start
        _, _, _, time_i_curr = incremental_svd(U, S, V, a, b)
        _, _, _, time_r_curr = restart_svd(U, S, V, del_A)
        time_t_curr = truncated_svd(X, s)
        # time_cg_curr = inverse_power_method(X)
        time_i += time_i_curr
        time_r += time_r_curr
        time_t += time_t_curr
        # time_cg += time_cg_curr
        X += del_A
    time_i_mean += time_i / len(sizes)
    time_r_mean += time_r / len(sizes)
    time_t_mean += time_t / len(sizes)
    time_svd_mean += time_svd / len(sizes)
    # time_cg_mean += time_cg / len(sizes)
print('Incremental SVD: {0}'.format(time_i_mean / len(trials)))
print('Restart SVD: {0}'.format(time_r_mean / len(trials)))
print('Truncated SVD: {0}'.format(time_t_mean / len(trials)))
print(' SVD: {0}'.format(time_svd_mean / len(trials)))
# print('CG: {0}'.format(time_cg_mean / len(trials)))
