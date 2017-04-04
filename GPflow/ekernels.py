from functools import reduce
import warnings
import tensorflow as tf
from . import kernels
from .tf_wraps import eye
from ._settings import settings
from .param import AutoFlow

from .quadrature import mvhermgauss
from numpy import pi as nppi

int_type = settings.dtypes.int_type
float_type = settings.dtypes.float_type

class SM(kernels.SM):

    def eKdiag(self, X, Xcov=None):
        """
        Also known as phi_0: <K_{x,x}>_{q(x)}
        :param X:
        :return: N
        """
        return self.Kdiag(X)

    def eKxz(self, Z, Xmu, Xcov):
        """
        Also known as phi_1: <K_{x, Z}>_{q(x)}.
        :param Z: MxD inducing inputs
        :param Xmu: X mean (NxD)
        :param Xcov: NxDxD
        :return: NxM
        """
        N, M = tf.shape(Xmu)[0], tf.shape(Z)[0]
        P = self.input_dim # equals D
        Q = self.Q

        lengthscales = tf.square(self.lengthscales)
        weights = self.weights
        frequencies = self.frequencies

        Sigma_q = 1./(4 * nppi**2) * tf.matrix_diag(1./lengthscales) # Q P P
        Sigma_q_det = tf.matrix_determinant(Sigma_q)
        Sigma_q_inv = (4 * nppi**2) * tf.matrix_diag(lengthscales) # Q P P
        Sigma_q_inv_ex = _expand_and_tile(Sigma_q_inv, 4, 0, N) # N Q P P

        S_inv = tf.matrix_inverse(Xcov) # N P P
        S_inv_ex = _expand_and_tile(S_inv, 4, 1, Q) # N Q P P

        A_q = tf.matrix_inverse(S_inv_ex + Sigma_q_inv_ex) # N Q P P
        tmp = _expand_and_tile(tf.matmul(A_q, S_inv_ex), 5, 1, M) # N M Q P P
        m = tf.matmul(tmp, _expand_and_tile(tf.expand_dims(_subs(Xmu,Z), -1), 5, 2, Q)) # N M Q P 1

        nu_q = tf.expand_dims(_expand_and_tile(2 * nppi * frequencies, 3, 0, N), -1) # N Q P 1
        nu_q_T = tf.transpose(nu_q, perm=[0,1,3,2]) # N Q 1 P

        exp = _expand_and_tile(tf.exp(-.5 * tf.matmul(tf.matmul(nu_q_T, A_q), nu_q)), 5, 1, M) # M M Q 1 1
        cos = tf.cos(tf.matmul(_expand_and_tile(nu_q_T, 5, 1, M), m)) # N M Q 1 1
        fac = weights * (2 * nppi)**(P/2.) * tf.sqrt(Sigma_q_det) # Q
        fac = _expand_and_tile(_expand_and_tile(fac, 2, 0, M), 3, 0, N) # N M Q

        # Calculation c: (2 pi)^(-k/2) |det(C)|^(-1/2) exp(-1/2 (x - mu)^T C^{-1} (x - mu))
        mean_c = tf.expand_dims(_expand_and_tile(_subs(Xmu,Z), 4, 2, Q), -1)  # N M Q P 1
        mean_c_T = tf.transpose(mean_c, perm=[0,1,2,4,3])
        tmp2 = _expand_and_tile(Sigma_q, 4, 0, N) + _expand_and_tile(Xcov, 4, 1, Q) # N Q P P
        var_c = _expand_and_tile(tmp2, 5, 1, M) # N M Q P P
        c_fac = 1./tf.sqrt(tf.matrix_determinant(var_c)) * (2 * nppi)**(-P/2.) # N M Q
        exp2 = tf.exp(-.5 * (tf.matmul(mean_c_T, tf.matmul(tf.matrix_inverse(var_c), mean_c)))) # N M Q 1 1
        c = c_fac * tf.squeeze(exp2, axis=[3,4])

        # Final
        res = tf.reduce_sum(c * fac * tf.squeeze(exp, axis=[3,4]) * tf.squeeze(cos, axis=[3,4]), axis= 2) # N M
        return res

    def eKzxKxz(self, Z, Xmu, Xcov):
        """
        Also known as Phi_2.
        :param Z: MxD
        :param Xmu: X mean (NxD)
        :param Xcov: X covariance matrices (NxDxD)
        :return: NxMxM
        """
        N, M = tf.shape(Xmu)[0], tf.shape(Z)[0]
        P = self.input_dim # equals D of the comment
        Q = self.Q

        lengthscales = tf.square(self.lengthscales)
        frequencies = 2 * nppi * self.frequencies
        weights = self.weights

        # C
        Sigma_q = 1./(4 * nppi**2) * tf.matrix_diag(1./lengthscales) # Q P P
        Sigma_q_det = tf.matrix_determinant(Sigma_q) # Q
        Sigma_q_det_cross = _prod(Sigma_q_det, Sigma_q_det) # Q Q
        weights_cross = _prod(weights, weights) # Q Q
        C = weights_cross * tf.sqrt(Sigma_q_det_cross) * (2*nppi)**P # Q Q
        C = expand_and_tile(expand_and_tile(expand_and_tile(C, 3, 0, M), 4, 0, M), 5, 0, N) # N M M Q Q


        # c1
        c1_var = expand_and_tile(expand_and_tile(_sum(Sigma_q, Sigma_q), 5, 0, M), 6, 0, M) # M M Q Q P P
        tau = tf.expand_dims(expand_and_tile(expand_and_tile(_subs(Z,Z), 4, 2, Q), 5, 2, Q), -1) # M M Q Q P 1
        tau_T = tf.transpose(tau, perm=[0, 1, 2, 3, 5, 4]) # M M Q Q 1 P
        c1_fac = 1./tf.sqrt(tf.matrix_determinant(c1_var)) * (2 * nppi)**(-P/2.) # M M Q Q
        c1_exp = tf.exp(-.5 * (tf.matmul(tau_T, tf.matmul(tf.matrix_inverse(c1_var), tau)))) # M M Q Q 1 1
        c1 = expand_and_tile(c1_fac * tf.squeeze(c1_exp, axis=[4,5]), 5, 0, N) # N M M Q Q

        # V1
        Sigma_q_inv = (4 * nppi**2) * tf.matrix_diag(lengthscales) # Q P P
        V1 = tf.matrix_inverse(_sum(Sigma_q_inv, Sigma_q_inv)) # Q Q P P
        V1_ext = expand_and_tile(expand_and_tile(V1, 5, 0, M), 6, 0, M) # M M Q Q P P

        # m1
        Sigma_q_inv_ext = expand_and_tile(Sigma_q_inv, 4, 0, M) # M Q P P
        y_ext = tf.expand_dims(expand_and_tile(Z, 3, 1, Q), -1) # M Q P 1
        tmp = tf.matmul(Sigma_q_inv_ext, y_ext) # M Q P 1
        tmp_ext1 = expand_and_tile(expand_and_tile(tmp, 5, 0, M), 6, 2, Q) # M M Q Q P 1
        tmp_ext2 = expand_and_tile(expand_and_tile(tmp, 5, 2, Q), 6, 1, M) # M M Q Q P 1
        m1 = tf.matmul(V1_ext, tmp_ext1 + tmp_ext2) # M M Q Q P 1

        # c2
        S_ext = expand_and_tile(expand_and_tile(expand_and_tile(expand_and_tile(Xcov, 4, 1, M), 5, 1, M), 6, 3, Q), 7, 3, Q) # N M M Q Q P P
        c2_var = expand_and_tile(V1_ext, 7, 0, N) + S_ext # N M M Q Q P P
        m1_ext = expand_and_tile(m1, 7, 0, N) # N M M Q Q P 1
        mu_ext = tf.expand_dims(expand_and_tile(expand_and_tile(expand_and_tile(expand_and_tile(Xmu, 3, 1, M), 4, 1, M), 5, 3, Q), 6, 3, Q), -1) # N M M Q Q P 1
        tau = m1_ext - mu_ext
        tau_T = tf.transpose(tau, perm=[0, 1, 2, 3, 4, 6, 5]) # N M M Q Q 1 P
        c2_fac = 1./tf.sqrt(tf.matrix_determinant(c2_var)) * (2 * nppi)**(-P/2.) # N M M Q Q
        c2_exp = tf.exp(-.5 * (tf.matmul(tau_T, tf.matmul(tf.matrix_inverse(c2_var), tau)))) # N M M Q Q 1 1
        c2 = c2_fac * tf.squeeze(c2_exp, axis=[5,6]) # N M M Q Q

        # V2
        V1_inv_ext = expand_and_tile(_sum(Sigma_q_inv, Sigma_q_inv), 5, 0, N)  # N Q Q P P
        S_inv = expand_and_tile(expand_and_tile(tf.matrix_inverse(Xcov), 4, 1, Q), 5, 1, Q)  # N Q Q P P
        V2 = expand_and_tile(expand_and_tile(tf.matrix_inverse(S_inv + V1_inv_ext), 6, 1, M), 7, 1, M) # N M M Q Q P P

        # m2
        V1_inv_ext = expand_and_tile(expand_and_tile(_sum(Sigma_q_inv, Sigma_q_inv), 5, 0, M), 6, 0, M)  # M M Q Q P P
        term1 = expand_and_tile(tf.matmul(V1_inv_ext, m1), 7, 0, N) # N M M Q Q P 1
        term2 = tf.matmul(tf.matrix_inverse(Xcov),tf.expand_dims(Xmu,-1)) # N P 1
        term2 = expand_and_tile(expand_and_tile(expand_and_tile(expand_and_tile(term2, 4, 1 , Q), 5, 1, Q), 6, 1, M), 7, 1, M) # N M M Q Q P 1
        m2 = tf.matmul(V2,term1 + term2) # N M M Q Q P 1

        # mean_alpha (ma)
        nu_cross = tf.transpose(_sum(tf.expand_dims(frequencies, -1), tf.expand_dims(frequencies, -1)), perm=[0, 1, 3, 2]) # Q Q 1 P
        nu_cross = expand_and_tile(expand_and_tile(expand_and_tile(nu_cross, 5, 0, M), 6, 0, M), 7, 0, N) # N M M Q Q 1 P
        tmp = tf.matmul(Z,tf.transpose(frequencies)) # M Q
        tmp_ext1 = expand_and_tile(expand_and_tile(tmp, 3, 0, M), 4, 2, Q) # M M Q Q
        tmp_ext2 = expand_and_tile(expand_and_tile(tmp, 3, -1, Q), 4, 1, M) # M M Q Q
        tmp_cross_min = -tmp_ext1 - tmp_ext2 # M M Q Q
        ma = tf.squeeze(tf.matmul(nu_cross, m2), axis=[5,6]) + expand_and_tile(tmp_cross_min, 5, 0, N) # N M M Q Q

        # mean_beta (mb)
        nu_cross_min = tf.transpose(_sum(tf.expand_dims(frequencies, -1), tf.expand_dims(-1. * frequencies, -1)), perm=[0, 1, 3, 2]) # Q Q 1 P
        nu_cross_min = expand_and_tile(expand_and_tile(expand_and_tile(nu_cross_min, 5, 0, M), 6, 0, M), 7, 0, N) # N M M Q Q 1 P
        tmp_cross_plus = tmp_ext1 - tmp_ext2 # M M Q Q
        mb = tf.squeeze(tf.matmul(nu_cross_min, m2), axis=[5,6]) + expand_and_tile(tmp_cross_plus, 5, 0, N) # N M M Q Q

        # sigma alpha (sa)
        sa = tf.squeeze(tf.matmul(tf.matmul(nu_cross, V2), nu_cross, transpose_b=True), axis=[5, 6]) # N M M Q Q

        # sigma alpha (sa)
        sb = tf.squeeze(tf.matmul(tf.matmul(nu_cross_min, V2), nu_cross_min, transpose_b=True), axis=[5, 6]) # N M M Q Q

        # Finally,
        res = .5 *  C * c1 * c2 * (tf.exp(-.5 * sa)*tf.cos(ma) + tf.exp(-.5 * sb)*tf.cos(mb)) # N M M Q Q
        res = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(res, axis=4), axis=3), axis=0) # M M

    def _expand_and_tile(self, tensor, rank, axis, multiple):
        """
        First expands the tensor in one dimension on the 'axis' position,
        then tiles the tensor 'multiple' times on 'axis'.

        Params:
        :tensor: tensor that will be expanded and tiled
        :rank: rank of the tensor after the operation
        :axis: axis to expand the tensor
        :multiple: number of times the tensor will be repeated

        Example:
            tensor A: with shape [2, 2]
            rank: tf.rank(A) + 1 = 3
            axis: 1
            multiple: 3

            returns: tf.tile(tf.expand_dims(A,axis), [1,multiple,1])
            The result is now rank 3 with shape [2, 3, 2]
        """
        return tf.tile(tf.expand_dims(tensor, axis), \
                       tf.stack([(multiple if ax == axis else 1) for ax in range(rank) ]))

    def _subs(self, X, X2):
        N, M = tf.shape(X)[0], tf.shape(X2)[0]
        X = tf.tile(tf.expand_dims(X, 1), [1, M, 1])
        X2 = tf.tile(tf.expand_dims(X2, 1), [1, N, 1])
        X2 = tf.transpose(X2, perm=[1, 0, 2])
        return tf.subtract(X,X2)

    def _prod(self, X, X2):
        N, M = tf.shape(X)[0], tf.shape(X2)[0]
        X = tf.tile(tf.expand_dims(X, 1), [1, M])
        X2 = tf.tile(tf.expand_dims(X2, 0), [N, 1])
        return X * X2

    def _sum(self, X, X2):
        N, M = tf.shape(X)[0], tf.shape(X2)[0]
        X = tf.tile(tf.expand_dims(X, 1), [1, M, 1, 1])
        X2 = tf.tile(tf.expand_dims(X2, 0), [N, 1, 1, 1])
        return X + X2


class RBF(kernels.RBF):
    def eKdiag(self, X, Xcov=None):
        """
        Also known as phi_0.
        :param X:
        :return: N
        """
        return self.Kdiag(X)

    def eKxz(self, Z, Xmu, Xcov):
        """
        Also known as phi_1: <K_{x, Z}>_{q(x)}.
        :param Z: MxD inducing inputs
        :param Xmu: X mean (NxD)
        :param Xcov: NxDxD
        :return: NxM
        """
        # use only active dimensions
        Xcov = self._slice_cov(Xcov)
        Z, Xmu = self._slice(Z, Xmu)
        D = tf.shape(Xmu)[1]
        lengthscales = self.lengthscales if self.ARD else tf.zeros((D,), dtype=float_type) + self.lengthscales

        vec = tf.expand_dims(Xmu, 2) - tf.expand_dims(tf.transpose(Z), 0)  # NxDxM
        chols = tf.cholesky(tf.expand_dims(tf.diag(lengthscales ** 2), 0) + Xcov)
        Lvec = tf.matrix_triangular_solve(chols, vec)
        q = tf.reduce_sum(Lvec ** 2, [1])

        chol_diags = tf.matrix_diag_part(chols)  # N x D
        half_log_dets = tf.reduce_sum(tf.log(chol_diags), 1) - tf.reduce_sum(tf.log(lengthscales))  # N,

        return self.variance * tf.exp(-0.5 * q - tf.expand_dims(half_log_dets, 1))

    def exKxz(self, Z, Xmu, Xcov):
        """
        <x_t K_{x_{t-1}, Z}>_q_{x_{t-1:t}}
        :param Z: MxD inducing inputs
        :param Xmu: X mean (N+1xD)
        :param Xcov: 2x(N+1)xDxD
        :return: NxMxD
        """
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(Xmu)[1], tf.constant(self.input_dim, dtype=int_type),
                            message="Currently cannot handle slicing in exKxz."),
            tf.assert_equal(tf.shape(Xmu), tf.shape(Xcov)[1:3], name="assert_Xmu_Xcov_shape")
        ]):
            Xmu = tf.identity(Xmu)

        M = tf.shape(Z)[0]
        N = tf.shape(Xmu)[0] - 1
        D = tf.shape(Xmu)[1]
        Xsigmb = tf.slice(Xcov, [0, 0, 0, 0], tf.stack([-1, N, -1, -1]))
        Xsigm = Xsigmb[0, :, :, :]  # NxDxD
        Xsigmc = Xsigmb[1, :, :, :]  # NxDxD
        Xmum = tf.slice(Xmu, [0, 0], tf.stack([N, -1]))
        Xmup = Xmu[1:, :]
        lengthscales = self.lengthscales if self.ARD else tf.zeros((D,), dtype=float_type) + self.lengthscales
        scalemat = tf.expand_dims(tf.diag(lengthscales ** 2.0), 0) + Xsigm  # NxDxD

        det = tf.matrix_determinant(
            tf.expand_dims(eye(tf.shape(Xmu)[1]), 0) + tf.reshape(lengthscales ** -2.0, (1, 1, -1)) * Xsigm
        )  # N

        vec = tf.expand_dims(tf.transpose(Z), 0) - tf.expand_dims(Xmum, 2)  # NxDxM
        smIvec = tf.matrix_solve(scalemat, vec)  # NxDxM
        q = tf.reduce_sum(smIvec * vec, [1])  # NxM

        addvec = tf.matmul(smIvec, Xsigmc, transpose_a=True) + tf.expand_dims(Xmup, 1)  # NxMxD

        return self.variance * addvec * tf.reshape(det ** -0.5, (N, 1, 1)) * tf.expand_dims(tf.exp(-0.5 * q), 2)

    def eKzxKxz(self, Z, Xmu, Xcov):
        """
        Also known as Phi_2.
        :param Z: MxD
        :param Xmu: X mean (NxD)
        :param Xcov: X covariance matrices (NxDxD)
        :return: NxMxM
        """
        # use only active dimensions
        Xcov = self._slice_cov(Xcov)
        Z, Xmu = self._slice(Z, Xmu)
        M = tf.shape(Z)[0]
        N = tf.shape(Xmu)[0]
        D = tf.shape(Xmu)[1]
        lengthscales = self.lengthscales if self.ARD else tf.zeros((D,), dtype=float_type) + self.lengthscales

        Kmms = tf.sqrt(self.K(Z, presliced=True)) / self.variance ** 0.5
        scalemat = tf.expand_dims(eye(D), 0) + 2 * Xcov * tf.reshape(lengthscales ** -2.0, [1, 1, -1])  # NxDxD
        det = tf.matrix_determinant(scalemat)

        mat = Xcov + 0.5 * tf.expand_dims(tf.diag(lengthscales ** 2.0), 0)  # NxDxD
        cm = tf.cholesky(mat)  # NxDxD
        vec = 0.5 * (tf.reshape(tf.transpose(Z), [1, D, 1, M]) +
                     tf.reshape(tf.transpose(Z), [1, D, M, 1])) - tf.reshape(Xmu, [N, D, 1, 1])  # NxDxMxM
        svec = tf.reshape(vec, (N, D, M * M))
        ssmI_z = tf.matrix_triangular_solve(cm, svec)  # NxDx(M*M)
        smI_z = tf.reshape(ssmI_z, (N, D, M, M)) # NxDxMxM
        fs = tf.reduce_sum(tf.square(smI_z), [1]) # NxMxM

        return self.variance ** 2.0 * tf.expand_dims(Kmms, 0) * tf.exp(-0.5 * fs) * tf.reshape(det ** -0.5, [N, 1, 1])


class Linear(kernels.Linear):
    def eKdiag(self, X, Xcov):
        if self.ARD:
            raise NotImplementedError
        # use only active dimensions
        X, _ = self._slice(X, None)
        Xcov = self._slice_cov(Xcov)
        return self.variance * (tf.reduce_sum(tf.square(X), 1) + tf.reduce_sum(tf.matrix_diag_part(Xcov), 1))

    def eKxz(self, Z, Xmu, Xcov):
        if self.ARD:
            raise NotImplementedError
        # use only active dimensions
        Z, Xmu = self._slice(Z, Xmu)
        return self.variance * tf.matmul(Xmu, tf.transpose(Z))

    def exKxz(self, Z, Xmu, Xcov):
        with tf.control_dependencies([
            tf.assert_equal(tf.shape(Xmu)[1], tf.constant(self.input_dim, int_type),
                            message="Currently cannot handle slicing in exKxz."),
            tf.assert_equal(tf.shape(Xmu), tf.shape(Xcov)[1:3], name="assert_Xmu_Xcov_shape")
        ]):
            Xmu = tf.identity(Xmu)

        N = tf.shape(Xmu)[0] - 1
        Xmum = Xmu[:-1, :]
        Xmup = Xmu[1:, :]
        op = tf.expand_dims(Xmum, 2) * tf.expand_dims(Xmup, 1) + Xcov[1, :-1, :, :]  # NxDxD
        return self.variance * tf.matmul(tf.tile(tf.expand_dims(Z, 0), (N, 1, 1)), op)

    def eKzxKxz(self, Z, Xmu, Xcov):
        """
        exKxz
        :param Z: MxD
        :param Xmu: NxD
        :param Xcov: NxDxD
        :return:
        """
        # use only active dimensions
        Xcov = self._slice_cov(Xcov)
        Z, Xmu = self._slice(Z, Xmu)
        N = tf.shape(Xmu)[0]
        mom2 = tf.expand_dims(Xmu, 1) * tf.expand_dims(Xmu, 2) + Xcov  # NxDxD
        eZ = tf.tile(tf.expand_dims(Z, 0), (N, 1, 1))  # NxMxD
        return self.variance ** 2.0 * tf.matmul(tf.matmul(eZ, mom2), eZ, transpose_b=True)


class Add(kernels.Add):
    """
    Add
    This version of Add will call the corresponding kernel expectations for each of the summed kernels. This will be
    much better for kernels with analytically calculated kernel expectations. If quadrature is to be used, it's probably
    better to do quadrature on the summed kernel function using `GPflow.kernels.Add` instead.
    """

    def __init__(self, kern_list):
        self.crossexp_funcs = {frozenset([Linear, RBF]): self.Linear_RBF_eKxzKzx}
        # self.crossexp_funcs = {}
        kernels.Add.__init__(self, kern_list)

    def eKdiag(self, X, Xcov):
        return reduce(tf.add, [k.eKdiag(X, Xcov) for k in self.kern_list])

    def eKxz(self, Z, Xmu, Xcov):
        return reduce(tf.add, [k.eKxz(Z, Xmu, Xcov) for k in self.kern_list])

    def exKxz(self, Z, Xmu, Xcov):
        return reduce(tf.add, [k.exKxz(Z, Xmu, Xcov) for k in self.kern_list])

    def eKzxKxz(self, Z, Xmu, Xcov):
        all_sum = reduce(tf.add, [k.eKzxKxz(Z, Xmu, Xcov) for k in self.kern_list])

        if self.on_separate_dimensions and Xcov.get_shape().ndims == 2:
            # If we're on separate dimensions and the covariances are diagonal, we don't need Cov[Kzx1Kxz2].
            crossmeans = []
            eKxzs = [k.eKxz(Z, Xmu, Xcov) for k in self.kern_list]
            for i, Ka in enumerate(eKxzs):
                for Kb in eKxzs[i + 1:]:
                    op = Ka[:, None, :] * Kb[:, :, None]
                    ct = tf.transpose(op, [0, 2, 1]) + op
                    crossmeans.append(ct)
            crossmean = reduce(tf.add, crossmeans)
            return all_sum + crossmean
        else:
            crossexps = []
            for i, ka in enumerate(self.kern_list):
                for kb in self.kern_list[i + 1:]:
                    try:
                        crossexp_func = self.crossexp_funcs[frozenset([type(ka), type(kb)])]
                        crossexp = crossexp_func(ka, kb, Z, Xmu, Xcov)
                    except (KeyError, NotImplementedError) as e:
                        print(str(e))
                        crossexp = self.quad_eKzx1Kxz2(ka, kb, Z, Xmu, Xcov)
                    crossexps.append(crossexp)
            return all_sum + reduce(tf.add, crossexps)

    def Linear_RBF_eKxzKzx(self, Ka, Kb, Z, Xmu, Xcov):
        Xcov = self._slice_cov(Xcov)
        Z, Xmu = self._slice(Z, Xmu)
        lin, rbf = (Ka, Kb) if type(Ka) is Linear else (Kb, Ka)
        assert type(lin) is Linear, "%s is not %s" % (str(type(lin)), str(Linear))
        assert type(rbf) is RBF, "%s is not %s" % (str(type(rbf)), str(RBF))
        if lin.ARD or type(lin.active_dims) is not slice or type(rbf.active_dims) is not slice:
            raise NotImplementedError("Active dims and/or Linear ARD not implemented. Switching to quadrature.")
        D = tf.shape(Xmu)[1]
        M = tf.shape(Z)[0]
        N = tf.shape(Xmu)[0]
        lengthscales = rbf.lengthscales if rbf.ARD else tf.zeros((D,), dtype=float_type) + rbf.lengthscales
        lengthscales2 = lengthscales ** 2.0

        const = rbf.variance * lin.variance * tf.reduce_prod(lengthscales)

        gaussmat = Xcov + tf.diag(lengthscales2)[None, :, :]  # NxDxD

        det = tf.matrix_determinant(gaussmat) ** -0.5  # N

        cgm = tf.cholesky(gaussmat)  # NxDxD
        tcgm = tf.tile(cgm[:, None, :, :], [1, M, 1, 1])
        vecmin = Z[None, :, :] - Xmu[:, None, :]  # NxMxD
        d = tf.matrix_triangular_solve(tcgm, vecmin[:, :, :, None])  # NxMxDx1
        exp = tf.exp(-0.5 * tf.reduce_sum(d ** 2.0, [2, 3]))  # NxM
        # exp = tf.Print(exp, [tf.shape(exp)])

        vecplus = (Z[None, :, :, None] / lengthscales2[None, None, :, None] +
                   tf.matrix_solve(Xcov, Xmu[:, :, None])[:, None, :, :])  # NxMxDx1
        mean = tf.cholesky_solve(tcgm,
                                 tf.matmul(tf.tile(Xcov[:, None, :, :], [1, M, 1, 1]), vecplus)
                                 )[:, :, :, 0] * lengthscales2[None, None, :]  # NxMxD
        a = tf.matmul(tf.tile(Z[None, :, :], [N, 1, 1]),
                            mean * exp[:, :, None] * det[:, None, None] * const, transpose_b=True)
        return a + tf.transpose(a, [0, 2, 1])

    def quad_eKzx1Kxz2(self, Ka, Kb, Z, Xmu, Xcov):
        # Quadrature for Cov[(Kzx1 - eKzx1)(kxz2 - eKxz2)]
        self._check_quadrature()
        warnings.warn("GPflow.ekernels.Add: Using numerical quadrature for kernel expectation cross terms.")
        Xmu, Z = self._slice(Xmu, Z)
        Xcov = self._slice_cov(Xcov)
        N, M, HpowD = tf.shape(Xmu)[0], tf.shape(Z)[0], self.num_gauss_hermite_points ** self.input_dim
        xn, wn = mvhermgauss(self.num_gauss_hermite_points, self.input_dim)

        # transform points based on Gaussian parameters
        cholXcov = tf.cholesky(Xcov)  # NxDxD
        Xt = tf.matmul(cholXcov, tf.tile(xn[None, :, :], (N, 1, 1)), transpose_b=True)  # NxDxH**D

        X = 2.0 ** 0.5 * Xt + tf.expand_dims(Xmu, 2)  # NxDxH**D
        Xr = tf.reshape(tf.transpose(X, [2, 0, 1]), (-1, self.input_dim))  # (H**D*N)xD

        cKa, cKb = [tf.reshape(
            k.K(tf.reshape(Xr, (-1, self.input_dim)), Z, presliced=False),
            (HpowD, N, M)
        ) - k.eKxz(Z, Xmu, Xcov)[None, :, :] for k in (Ka, Kb)]  # Centred Kxz
        eKa, eKb = Ka.eKxz(Z, Xmu, Xcov), Kb.eKxz(Z, Xmu, Xcov)

        wr = wn * nppi ** (-self.input_dim * 0.5)
        cc = tf.reduce_sum(cKa[:, :, None, :] * cKb[:, :, :, None] * wr[:, None, None, None], 0)
        cm = eKa[:, None, :] * eKb[:, :, None]
        return cc + tf.transpose(cc, [0, 2, 1]) + cm + tf.transpose(cm, [0, 2, 1])


class Prod(kernels.Prod):
    def eKdiag(self, Xmu, Xcov):
        if not self.on_separate_dimensions:
            raise NotImplementedError("Prod currently needs to be defined on separate dimensions.")  # pragma: no cover
        with tf.control_dependencies([
            tf.assert_equal(tf.rank(Xcov), 2,
                            message="Prod currently only supports diagonal Xcov.", name="assert_Xcov_diag"),
        ]):
            return reduce(tf.multiply, [k.eKdiag(Xmu, Xcov) for k in self.kern_list])

    def eKxz(self, Z, Xmu, Xcov):
        if not self.on_separate_dimensions:
            raise NotImplementedError("Prod currently needs to be defined on separate dimensions.")  # pragma: no cover
        with tf.control_dependencies([
            tf.assert_equal(tf.rank(Xcov), 2,
                            message="Prod currently only supports diagonal Xcov.", name="assert_Xcov_diag"),
        ]):
            return reduce(tf.multiply, [k.eKxz(Z, Xmu, Xcov) for k in self.kern_list])

    def eKzxKxz(self, Z, Xmu, Xcov):
        if not self.on_separate_dimensions:
            raise NotImplementedError("Prod currently needs to be defined on separate dimensions.")  # pragma: no cover
        with tf.control_dependencies([
            tf.assert_equal(tf.rank(Xcov), 2,
                            message="Prod currently only supports diagonal Xcov.", name="assert_Xcov_diag"),
        ]):
            return reduce(tf.multiply, [k.eKzxKxz(Z, Xmu, Xcov) for k in self.kern_list])
