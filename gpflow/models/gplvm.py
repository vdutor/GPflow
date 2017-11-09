import tensorflow as tf
import numpy as np

from gpflow import settings
from gpflow import likelihoods
from gpflow import transforms
from gpflow import kernels
from gpflow import conditionals
from gpflow import kullback_leiblers
from gpflow import features

from gpflow.models.model import GPModel
from gpflow.models.gpr import GPR
from gpflow.params import Parameter, DataHolder, Minibatch
from gpflow.decors import params_as_tensors
from gpflow.mean_functions import Zero
from gpflow.quadrature import mvnquad


class GPLVM(GPR):
    """
    Standard GPLVM where the likelihood can be optimised with respect to the latent X.
    """

    def __init__(self, Y, latent_dim, X_mean=None, kern=None, mean_function=Zero(), **kwargs):
        """
        Initialise GPLVM object. This method only works with a Gaussian likelihood.
        :param Y: data matrix, size N (number of points) x D (dimensions)
        :param Z: matrix of inducing points, size M (inducing points) x Q (latent dimensions)
        :param X_mean: latent positions (N x Q), for the initialisation of the latent space.
        :param kern: kernel specification, by default RBF
        :param mean_function: mean function, by default None.
        """
        if kern is None:
            kern = kernels.RBF(latent_dim, ARD=True)
        if X_mean is None:
            X_mean = PCA_reduce(Y, latent_dim)
        num_latent = X_mean.shape[1]
        if num_latent != latent_dim:
            msg = 'Passed in number of latent {0} does not match initial X {1}.'
            raise ValueError(msg.format(latent_dim, num_latent))
        if Y.shape[1] < num_latent:
            raise ValueError('More latent dimensions than observed.')
        GPR.__init__(self, X_mean, Y, kern, mean_function=mean_function, **kwargs)
        del self.X  # in GPLVM this is a Param
        self.X = Parameter(X_mean)


class BayesianGPLVM(GPModel):
    def __init__(self, X_mean, X_var, Y, kern, M, Z=None, X_prior_mean=None, X_prior_var=None):
        """
        Initialise Bayesian GPLVM object. This method only works with a Gaussian likelihood.
        :param X_mean: initial latent positions, size N (number of points) x Q (latent dimensions).
        :param X_var: variance of latent positions (N x Q), for the initialisation of the latent space.
        :param Y: data matrix, size N (number of points) x D (dimensions)
        :param kern: kernel specification, by default RBF
        :param M: number of inducing points
        :param Z: matrix of inducing points, size M (inducing points) x Q (latent dimensions). By default
        random permutation of X_mean.
        :param X_prior_mean: prior mean used in KL term of bound. By default 0. Same size as X_mean.
        :param X_prior_var: pripor variance used in KL term of bound. By default 1.
        """
        GPModel.__init__(self, X_mean, Y, kern, likelihood=likelihoods.Gaussian(), mean_function=Zero())
        del self.X  # in GPLVM this is a Param
        self.X_mean = Parameter(X_mean)
        # diag_transform = transforms.DiagMatrix(X_var.shape[1])
        # self.X_var = Parameter(diag_transform.forward(transforms.positive.backward(X_var)) if X_var.ndim == 2 else X_var,
        #                    diag_transform)
        assert X_var.ndim == 2
        self.X_var = Parameter(X_var, transform=transforms.positive)
        self.num_data = X_mean.shape[0]
        self.output_dim = Y.shape[1]

        assert np.all(X_mean.shape == X_var.shape)
        assert X_mean.shape[0] == Y.shape[0], 'X mean and Y must be same size.'
        assert X_var.shape[0] == Y.shape[0], 'X var and Y must be same size.'

        # inducing points
        if Z is None:
            # By default we initialize by subset of initial latent points
            Z = np.random.permutation(X_mean.copy())[:M]
        else:
            assert Z.shape[0] == M
        self.Z = Parameter(Z)
        self.num_latent = Z.shape[1]
        assert X_mean.shape[1] == self.num_latent

        # deal with parameters for the prior mean variance of X
        if X_prior_mean is None:
            X_prior_mean = np.zeros((self.num_data, self.num_latent))
        if X_prior_var is None:
            X_prior_var = np.ones((self.num_data, self.num_latent))

        self.X_prior_mean = np.asarray(np.atleast_1d(X_prior_mean), dtype=settings.np_float)
        self.X_prior_var = np.asarray(np.atleast_1d(X_prior_var), dtype=settings.np_float)

        assert self.X_prior_mean.shape[0] == self.num_data
        assert self.X_prior_mean.shape[1] == self.num_latent
        assert self.X_prior_var.shape[0] == self.num_data
        assert self.X_prior_var.shape[1] == self.num_latent

    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        num_inducing = tf.shape(self.Z)[0]
        psi0 = tf.reduce_sum(self.kern.eKdiag(self.X_mean, self.X_var), 0)
        psi1 = self.kern.eKxz(self.Z, self.X_mean, self.X_var)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, self.X_mean, self.X_var), 0)
        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=settings.tf_float) * settings.numerics.jitter_level
        L = tf.cholesky(Kuu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=settings.tf_float)
        LB = tf.cholesky(B)
        log_det_B = 2. * tf.reduce_sum(tf.log(tf.matrix_diag_part(LB)))
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma

        # KL[q(x) || p(x)]
        dX_var = self.X_var if len(self.X_var.get_shape()) == 2 else tf.matrix_diag_part(self.X_var)
        NQ = tf.cast(tf.size(self.X_mean), settings.tf_float)
        D = tf.cast(tf.shape(self.Y)[1], settings.tf_float)
        KL = -0.5 * tf.reduce_sum(tf.log(dX_var)) \
             + 0.5 * tf.reduce_sum(tf.log(self.X_prior_var)) \
             - 0.5 * NQ \
             + 0.5 * tf.reduce_sum((tf.square(self.X_mean - self.X_prior_mean) + dX_var) / self.X_prior_var)

        # compute log marginal bound
        ND = tf.cast(tf.size(self.Y), settings.tf_float)
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(self.Y)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 -
                             tf.reduce_sum(tf.matrix_diag_part(AAT)))
        bound -= KL
        return bound

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points.
        Note that this is very similar to the SGPR prediction, for which
        there are notes in the SGPR notebook.
        :param Xnew: Point to predict at.
        """
        num_inducing = tf.shape(self.Z)[0]
        psi1 = self.kern.eKxz(self.Z, self.X_mean, self.X_var)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, self.X_mean, self.X_var), 0)
        Kuu = self.kern.K(self.Z) + tf.eye(num_inducing, dtype=settings.tf_float) * settings.numerics.jitter_level
        Kus = self.kern.K(self.Z, Xnew)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.cholesky(Kuu)

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=settings.tf_float)
        LB = tf.cholesky(B)
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma
        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = self.kern.K(Xnew) + tf.matmul(tmp2, tmp2, transpose_a=True) \
                  - tf.matmul(tmp1, tmp1, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(Xnew) + tf.reduce_sum(tf.square(tmp2), 0) \
                  - tf.reduce_sum(tf.square(tmp1), 0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew), var


class BayesianGPLVM_Optimal_qX(GPModel):

    def __init__(self, latent_dim, Y, kern, feat, *,
                    H = 15,
                    Z = None,
                    X_prior_mean=None,
                    X_prior_var=None,
                    minibatch_size=None,
                    whiten=True,
                    q_diag = False,
                    mean_function=Zero()):

        num_data = Y.shape[0]

        if X_prior_mean is None:
            X_prior_mean = np.zeros((num_data, latent_dim))
        if X_prior_var is None:
            X_prior_var = np.ones((num_data, latent_dim))

        X = np.concatenate([X_prior_mean, X_prior_var], axis=1)
        print(X.shape)

        if minibatch_size is None:
            minibatch_size = num_data
            X = DataHolder(X)
            Y = DataHolder(Y)
        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=0)

        # init the super class, accept args
        GPModel.__init__(self, X, Y, kern, likelihoods.Gaussian(), mean_function)

        self.X_prior_mean = X_prior_mean
        self.X_prior_var = X_prior_var

        self.minibatch_size = minibatch_size
        self.latent_dim = latent_dim
        self.num_data = Y.shape[0]
        self.num_func = Y.shape[1]
        self.H = H

        self.whiten = whiten
        self.q_diag = q_diag
        self.feat = features.inducingpoint_wrapper(feat, Z)
        self.num_inducing = len(self.feat)

        # init variational parameters
        self.q_mu = Parameter(np.zeros((self.num_inducing, self.num_func), dtype=settings.np_float))
        if self.q_diag:
            self.q_sqrt = Parameter(np.ones((self.num_inducing, self.num_func), dtype=settings.np_float), transforms.positive)
        else:
            q_sqrt = np.array([np.eye(self.num_inducing, dtype=settings.np_float) for _ in range(self.num_func)]).swapaxes(0, 2)
            self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(self.num_inducing, self.num_func))

    @params_as_tensors
    def _build_likelihood(self):
        X_prior_mean = self.X[:, :self.latent_dim]
        X_prior_var = tf.matrix_diag(self.X[:, -self.latent_dim:])

        minibatch_size_float = tf.cast(self.minibatch_size, dtype=settings.tf_float)
        scale = tf.cast(self.num_data, dtype=settings.tf_float) / minibatch_size_float

        KL_u = self.build_prior_KL_qU()

        def expected_log_likelihood(X):
            Y_tiled = tf.tile(self.Y[None, :, :], [self.H**self.latent_dim, 1, 1])
            Y_reshaped = tf.reshape(Y_tiled, [-1, self.num_func])
            q_f_mean, q_f_var = conditionals.feature_conditional(X, self.feat, self.kern, self.q_mu, q_sqrt=self.q_sqrt, whiten=self.whiten)
            psi = tf.reduce_sum(self.likelihood.variational_expectations(q_f_mean, q_f_var, Y_reshaped), axis=1)
            return tf.exp(psi)

        Zn = mvnquad(expected_log_likelihood, X_prior_mean, X_prior_var ** 0.5, self.H, self.latent_dim, Dout=(1,), chol=True)
        log_Z = tf.reduce_sum(tf.log(Zn), axis=0)

        return - log_Z * scale - KL_u

    @params_as_tensors
    def build_prior_KL_qU(self):
        if self.whiten:
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_white_diag(self.q_mu, self.q_sqrt)
            else:
                KL = kullback_leiblers.gauss_kl_white(self.q_mu, self.q_sqrt)
        else:
            K = self.feat.Kuu(self.kern, jitter=settings.numerics.jitter_level)
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_diag(self.q_mu, self.q_sqrt, K)
            else:
                KL = kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)
        return KL

def PCA_reduce(X, Q):
    """
    A helpful function for linearly reducing the dimensionality of the data X
    to Q.
    :param X: data array of size N (number of points) x D (dimensions)
    :param Q: Number of latent dimensions, Q < D
    :return: PCA projection array of size N x Q.
    """
    assert Q <= X.shape[1], 'Cannot have more latent dimensions than observed'
    evecs, evals = np.linalg.eigh(np.cov(X.T))
    i = np.argsort(evecs)[::-1]
    W = evals[:, i]
    W = W[:, :Q]
    return (X - X.mean(0)).dot(W)
