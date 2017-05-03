# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, fujiisoup
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
import tensorflow as tf
from .model import GPModel
from .densities import multivariate_normal
from .mean_functions import Zero, Constant
from . import likelihoods
from .param import DataHolder, AutoFlow
from ._settings import settings
float_type = settings.dtypes.float_type


class GPR(GPModel):
    """
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood i this models is sometimes referred to as the 'marginal log likelihood', and is given by

    .. math::

       \\log p(\\mathbf y \\,|\\, \\mathbf f) = \\mathcal N\\left(\\mathbf y\,|\, 0, \\mathbf K + \\sigma_n \\mathbf I\\right)
    """
    def __init__(self, X, Y, kern, mean_function=Zero(), likelihood=likelihoods.Gaussian(), name='name'):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        # likelihood = likelihoods.Gaussian()
        X = DataHolder(X, on_shape_change='pass')
        Y = DataHolder(Y, on_shape_change='pass')
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, name)
        self.num_latent = Y.shape[1]

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=float_type) * self.likelihood.variance
        L = tf.cholesky(K)
        m = self.mean_function(self.X)

        return multivariate_normal(self.Y, m, L)

    def build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=float_type) * self.likelihood.variance
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(self.X))
        fmean = tf.matmul(A, V, transpose_a=True) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar


    def build_predict_uncertain(self, Xmean, Xvar):
        Ns = tf.shape(Xmean)[0]
        N = tf.shape(self.X)[0]
        D = tf.shape(self.Y)[1]

        psi0 = self.kern.eKdiag(Xmean, Xvar) # Ns
        psi1 = self.kern.eKxz(self.X, Xmean, Xvar) # Ns x N
        psi2 = self.kern.eKzxKxz(self.X, Xmean, Xvar) # Ns x N x N

        K = self.kern.K(self.X)
        K += tf.eye(tf.shape(self.X)[0], dtype=float_type) * self.likelihood.variance # N x N
        L = tf.cholesky(K) # N x N
        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) # N x Ns
        V = tf.matrix_triangular_solve(L, self.Y) # N x D
        fmean = tf.matmul(tf.transpose(A), V) # Ns x D

        # Following nknudde's derivation: a = V
        L3 = tf.tile(tf.expand_dims(L, 0), [tf.shape(Xmean)[0], 1, 1]) # Ns x N x N
        b = tf.matrix_triangular_solve(L3, tf.transpose(tf.matrix_triangular_solve(L3, psi2), perm=[0, 2, 1])) # Ns x N x N

        # In case the capabilities of einsum would expand, the following lines can be replaced by
        # TT = tf.einsum('ij,ljk,if->li', V, b, V)
        tmp1 = tf.transpose(tf.reshape(tf.matmul(tf.reshape(b, [Ns*N, N]), V), [Ns, N, D]), perm=[0,2,1]) # Ns x D x N
        tmp2 = tf.reshape(tf.matmul(tf.reshape(tmp1, [Ns*D, N]), V), [Ns, D, D]) # Ns x D x D
        TT = tf.transpose(tf.matrix_diag_part(tmp2)) - tf.trace(b) # D x Ns
        fvar = tf.transpose(psi0 + TT - tf.transpose(tf.square(fmean)))
        return fmean, fvar


    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def predict_f_uncertain(self, Xstarmu, Xstarvar):
        """
        Predicts the first and second moment of the Gaussian distribution, matched to the distribution obtained by
        by propagating a Gaussian distribution.
        Note: this method is only available in combination with Constant or Zero mean functions.
        :param Xstarmu: mean of the points in latent space size: Nnew (number of new points ) x Q (latent dim)
        :param Xstarvar: variance of the points in latent space size: Nnew (number of new points ) x Q (latent dim)
        :returns (mean, covar)
        :rtype mean: np.ndarray, size Nnew (number of new points ) x D
        covar: np.ndarray, size Nnew (number of new points ) x D x D
        """
        assert (isinstance(self.mean_function, (Zero)))
        return self.build_predict_uncertain(Xstarmu, Xstarvar)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def predict_y_uncertain(self, Xstarmu, Xstarvar):
        """
        Predicts the first and second moment of the Gaussian distribution, matched to the distribution obtained by
        by propagating a Gaussian distribution.
        Note: this method is only available in combination with Constant or Zero mean functions.
        """
        assert (isinstance(self.mean_function, (Zero)))
        mean, covar = self.build_predict_uncertain(Xstarmu, Xstarvar)
        return self.likelihood.predict_mean_and_var(mean, covar)
