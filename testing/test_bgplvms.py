import numpy as np

from gpflow.kernels import RBF
from gpflow.models import BayesianGPLVM_Optimal_qX
from gpflow.training import ScipyOptimizer


N = 1000
M = 20
D = 5
Q = 2
Y = np.random.randn(N, D)
Z = np.random.randn(M, Q)

m = BayesianGPLVM_Optimal_qX(Q, Y, RBF(Q), Z, minibatch_size=50, whiten=True, q_diag=True)
m.compile()

opt = ScipyOptimizer()

print(m.compute_log_likelihood())
opt.minimize(m)
print(m.compute_log_likelihood())

m = BayesianGPLVM_Optimal_qX(Q, Y, RBF(Q), Z, minibatch_size=50, whiten=True, q_diag=True)
