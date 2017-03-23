from functools import reduce
import numpy as np
import tensorflow as tf
from .tf_wraps import eye
from .param import Parameterized, Param, AutoFlow, ParamList
from .conditionals import conditional
from .kullback_leiblers import gauss_kl
from .model import Model
from . import transforms
from . import ekernels
from ._settings import settings

float_type = settings.dtypes.float_type

def cho_solve(L, X):
    return tf.matrix_triangular_solve(tf.transpose(L),
                                      tf.matrix_triangular_solve(L, X), lower=False)


class Layer(Parameterized):
    def __init__(self, input_dim, output_dim, kern, Z, beta=10.0):
        super(Layer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_inducing = Z.shape[0]

        assert Z.shape[1] == self.input_dim
        self.kern = kern
        self.Z = Param(Z)
        self.beta = Param(beta, transforms.positive)

        shape = (self.num_inducing, self.output_dim)
        self.q_mu = Param(np.zeros(shape))
        q_sqrt = np.array([np.eye(self.num_inducing)
                           for _ in range(self.output_dim)]).swapaxes(0, 2)  # M x M x D
        self.q_sqrt = Param(q_sqrt)

    def build_kl(self, Kmm):
        return gauss_kl(self.q_mu, self.q_sqrt, Kmm)

    def build_predict(self, Xnew, full_cov=False):
        return conditional(Xnew, self.Z, self.kern,
                           self.q_mu, full_cov=full_cov,
                           q_sqrt=self.q_sqrt, whiten=False)

    @AutoFlow((float_type, [None, None]))
    def predict_f(self, X):
        return self.build_predict(X)

    @AutoFlow((float_type, [None, None]))
    def predict_f_samples(self, X):
        return self.build_posterior_samples(X)

    def build_posterior_samples(self, Xtest, full_cov=False):
        m, v = self.build_predict(Xtest, full_cov=full_cov)
        if full_cov:
            samples = []
            for i in range(self.output_dim):
                L = tf.cholesky(v[:, :, i] + eye(tf.shape(v)[1]) / self.beta)
                W = tf.random_normal(tf.pack([tf.shape(m)[0], 1]), dtype=float_type)
                samples.append(m[:, i:i+1] + tf.matmul(L, W))
            return tf.concat(1, samples)
        else:
            return m + tf.random_normal(tf.shape(m), dtype=float_type)*tf.sqrt(v + 1./self.beta)


class HiddenLayer(Layer):

    def feed_forward(self, X_in_mean, X_in_var):
        """
        Compute the variational distribution for the outputs of this layer, as
        well as any marginal likelihood terms that occur
        """
        # kernel computations
        psi0 = tf.reduce_sum(self.kern.eKdiag(X_in_mean, X_in_var))
        psi1 = self.kern.eKxz(self.Z, X_in_mean, X_in_var) # N x M
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, X_in_mean, X_in_var), 0)  # M x M

        Kmm = self.kern.K(self.Z) + np.eye(self.num_inducing)*settings.numerics.jitter_level # M x M
        Lmm = tf.cholesky(Kmm) # M x M

        # useful computations
        KmmiPsi2 = cho_solve(Lmm, psi2) # M x M
        # = chol of latent variances (M x M x D),
        # D x M x M
        q_chol = tf.matrix_band_part(tf.transpose(self.q_sqrt, (2, 0, 1)), -1, 0) # D x M x M
        q_cov = tf.matmul(q_chol, tf.transpose(q_chol, perm=[0, 2, 1]))  # D x M x M
        uuT = tf.matmul(self.q_mu, tf.transpose(self.q_mu)) + tf.reduce_sum(q_cov, 0) # M x M

        # trace term, KL
        trace = psi0 - tf.reduce_sum(tf.diag_part(KmmiPsi2))
        self._log_marginal_contribution = -0.5 * self.beta * self.output_dim * trace
        self._log_marginal_contribution -= self.build_kl(Kmm)

        # distribution to feed forward to downstream layers
        psi1Kmmi = tf.transpose(cho_solve(Lmm, tf.transpose(psi1))) # N x M
        forward_mean = tf.matmul(psi1Kmmi, self.q_mu) # N x D
        # Here: D x N x M MULT D X M X M
        tmp = tf.einsum('ij,kjl->ikl', psi1Kmmi, q_chol) # N x D x M
        forward_var = tf.matmul(tmp, tf.transpose(tmp, perm=[0,2,1])) + eye(tf.shape(q_chol)[0]) / self.beta  # N x D x D

        # complete the square term
        KmmiPsi2Kmmi = cho_solve(Lmm, tf.transpose(KmmiPsi2)) #
        tmp = KmmiPsi2Kmmi - tf.matmul(tf.transpose(psi1Kmmi), psi1Kmmi)
        self._log_marginal_contribution += -0.5 * self.beta * tf.reduce_sum(tmp * uuT)

        return forward_mean, forward_var


class InputLayerFixed(Layer):
    def __init__(self, X, input_dim, output_dim, kern, Z, beta=500.):
        super(InputLayerFixed, self).__init__(input_dim=input_dim, output_dim=output_dim, kern=kern, Z=Z, beta=beta)
        self.X = X

    def feed_forward(self):
        # kernel computations
        kdiag = self.kern.Kdiag(self.X)
        Knm = self.kern.K(self.X, self.Z)
        Kmm = self.kern.K(self.Z) + eye(self.num_inducing)*settings.numerics.jitter_level
        Lmm = tf.cholesky(Kmm)
        A = tf.matrix_triangular_solve(Lmm, tf.transpose(Knm))

        # trace term, KL term
        trace = tf.reduce_sum(kdiag) - tf.reduce_sum(tf.square(A))
        self._log_marginal_contribution = -0.5*self.beta*self.output_dim * trace
        self._log_marginal_contribution -= self.build_kl(Kmm)

        # feed outputs to next layer
        mu, var = self.build_predict(self.X, full_cov=False)
        return mu, tf.matrix_diag(var) + 1./self.beta


class ObservedLayer(Layer):
    def __init__(self, Y, input_dim, output_dim, kern, Z, beta=0.01):
        super(ObservedLayer, self).__init__(input_dim=input_dim, output_dim=output_dim, kern=kern, Z=Z, beta=beta)
        assert Y.shape[1] == output_dim
        self.Y = Y

    def feed_forward(self, X_in_mean, X_in_var):
        # kernel computations
        psi0 = tf.reduce_sum(self.kern.eKdiag(X_in_mean, X_in_var))
        psi1 = self.kern.eKxz(self.Z, X_in_mean, X_in_var) # N x M
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, X_in_mean, X_in_var), 0)

        Kmm = self.kern.K(self.Z) + eye(self.num_inducing)*settings.numerics.jitter_level
        Lmm = tf.cholesky(Kmm)
        q_chol = tf.matrix_band_part(tf.transpose(self.q_sqrt, (2, 0, 1)), -1, 0)  # force lower triangle
        q_cov = tf.matmul(q_chol, tf.transpose(q_chol, perm=[0, 2, 1]))  # D x M x M
        uuT = tf.matmul(self.q_mu, tf.transpose(self.q_mu)) + tf.reduce_sum(q_cov, 0)

        # trace term
        KmmiPsi2 = cho_solve(Lmm, psi2)
        trace = psi0 - tf.reduce_sum(tf.diag_part(KmmiPsi2))
        self._log_marginal_contribution = -0.5 * self.beta * self.output_dim * trace

        # CTS term -- including Thang's correction
        KmmiuuT = cho_solve(Lmm, uuT)
        self._log_marginal_contribution += -0.5 * self.beta * tf.reduce_sum(KmmiPsi2 * tf.transpose(KmmiuuT))

        # KL term
        self._log_marginal_contribution -= self.build_kl(Kmm)

        # data fit terms
        A = tf.transpose(cho_solve(Lmm, tf.transpose(psi1)))
        proj_mean = tf.matmul(A, self.q_mu)
        N = tf.cast(tf.shape(X_in_mean)[0], float_type)
        self._log_marginal_contribution += -0.5 * N * self.output_dim * tf.log(2 * np.pi / self.beta)
        self._log_marginal_contribution += -0.5 * self.beta * (np.sum(np.square(self.Y)) -
                                                               2.*tf.reduce_sum(self.Y*proj_mean))

    def build_predict_uncertain(self, X_in_mean, X_in_var):
        psi1 = self.kern.eKxz(self.Z, X_in_mean, X_in_var)
        Kmm = self.kern.K(self.Z) + np.eye(self.num_inducing) * settings.numerics.jitter_level
        Lmm = tf.cholesky(Kmm)
        q_chol = tf.matrix_band_part(tf.transpose(self.q_sqrt, (2, 0, 1)), -1, 0)  # force lower triangle
        psi1Kmmi = tf.transpose(cho_solve(Lmm, tf.transpose(psi1)))
        forward_mean = tf.matmul(psi1Kmmi, self.q_mu)
        tmp = tf.einsum('ij,kjl->ikl', psi1Kmmi, q_chol)  # N x D x M
        forward_var = tf.reduce_sum(tf.square(tmp), 2) + 1. / self.beta
        return forward_mean, forward_var


    def build_posterior_samples(self, Xtest, full_cov=False):
        """
        in the special case of the last layer, don't add noise to the predicted
        samples (we want to predict the underlying value instead...)
        """
        m, v = self.build_predict(Xtest, full_cov=full_cov)
        if full_cov:
            samples = []
            for i in range(self.output_dim):
                L = tf.cholesky(v[:, :, i] + eye(tf.shape(v)[1]) * settings.numerics.jitter_level)
                W = tf.random_normal(tf.pack([tf.shape(m)[0], 1]), dtype=float_type)
                samples.append(m[:, i:i+1] + tf.matmul(L, W))
            return tf.concat(1, samples)
        else:
            return m + tf.random_normal(tf.shape(m), dtype=float_type)*tf.sqrt(v)


class ColDeep(Model):
    def __init__(self, X, Y, Qs, Ms, ARD_X=False):
        """
        Build a coldeep structure with len(Qs) hidden layers.

        Note that len(Ms) = len(Qs) + 1, since there's always 1 more GP than there
        are hidden layers.
        """

        super(ColDeep, self).__init__()
        assert len(Ms) == (1 + len(Qs))

        Nx, D_in = X.shape
        Ny, D_out = Y.shape
        assert Nx == Ny
        self.layers = ParamList([])

        # input layer
        Z0 = np.linspace(0, 1, Ms[0]).reshape(-1, 1) * (X.max(0)-X.min(0)) + X.min(0)
        self.layers.append(InputLayerFixed(X=X,
                           input_dim=D_in,
                           output_dim=Qs[0],
                           kern=ekernels.RBF(D_in, ARD=ARD_X),
                           Z=Z0,
                           beta=100.))
        # hidden layers
        for h in range(len(Qs)-1):
            Z0 = np.tile(np.linspace(-3, 3, Ms[h+1]).reshape(-1, 1), [1, Qs[h]])
            self.layers.append(HiddenLayer(input_dim=Qs[h],
                               output_dim=Qs[h+1],
                               kern=ekernels.RBF(Qs[h], ARD=ARD_X),
                               Z=Z0,
                               beta=100.))
        # output layer
        Z0 = np.tile(np.linspace(-3, 3, Ms[-1]).reshape(-1, 1), [1, Qs[-1]])
        self.layers.append(ObservedLayer(Y=Y,
                           input_dim=Qs[-1],
                           output_dim=D_out,
                           kern=ekernels.RBF(Qs[-1], ARD=ARD_X),
                           Z=Z0,
                           beta=500.))

    def build_likelihood(self):
        mu, var = self.layers[0].feed_forward()
        for l in self.layers[1:-1]:
            mu, var = l.feed_forward(mu, var)
        self.layers[-1].feed_forward(mu, var)
        return reduce(tf.add, [l._log_marginal_contribution for l in self.layers])

    @AutoFlow((float_type,[None, None]))
    def predict_f(self, Xnew):
        mu, var = self.layers[0].build_predict(Xnew)
        # First GP mapping results in i.i.d predictions. -> turn into diagonal matrix
        var = tf.matrix_diag(var)
        for l in self.layers[1:-1]:
            mu, var = l.feed_forward(mu, var)
        return self.layers[-1].build_predict_uncertain(mu, var)

    @AutoFlow((float_type,))
    def predict_sampling(self, Xtest):
        for l in self.layers:
            Xtest = l.build_posterior_samples(Xtest, full_cov=False)
        return Xtest

    @AutoFlow((float_type,))
    def predict_sampling_correlated(self, Xtest):
        for l in self.layers:
            Xtest = l.build_posterior_samples(Xtest, full_cov=True)
        return Xtest