# -*- coding: utf-8 -*-

import logging

import numpy as np

from timeit import default_timer
from typing import Sequence, Optional

from shgo import shgo

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import pairwise_distances, _VALID_METRICS

from numpy.linalg import multi_dot

from scipy import linalg

from .kernels import *

__all__ = [
    'OptimizedKRR'
]

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s\n\r%(message)s',
    datefmt='%H:%M:%S'
)


class OptimizedKRR(RegressorMixin, BaseEstimator):
    def __init__(
            self,
            init_sigma: float = None,
            init_alpha: float = 1e-3,
            ker_type: str = 'gau',
            ker_pow: float = None,
            pd_metric: str = 'euclidean',
            pd_metric_kwargs: dict = None,
            pd_calc_n_jobs: int = 8,
            distance_factor: float = 8.0
    ):
        """
        # TODO write description

        Base class for KRR with optimization

        :param init_sigma: initial kernel width
        :param init_alpha: initial regularization parameter
        :param ker_type: kernel functional
        :param ker_pow: power of a generic kernel
        :param pd_metric: pairwise distance metric
        :param pd_metric_kwargs: pairwise distance metric kwargs
        :param pd_calc_n_jobs: n_jobs for pairwise distances' calculation
        :param distance_factor: we assume that optimization bounds for sigma are proportional
                                to min and max distances in the dataset multiplied by some factor.
                                Decreasing this factor (but make sure it's >=1) speeds up optimization.
        """
        if init_alpha <= 0:
            raise ValueError(f'Regularization parameter alpha must be greater than zero, got {init_alpha}.')
        if ker_type not in VALID_KERNELS:
            raise ValueError(f'Wrong kernel function name, please choose from: {VALID_KERNELS}.')
        if ker_type == 'gen' and ker_pow is None:
            raise ValueError('Argument "ker_pow" must be provided when using the "gen" kernel.')
        if distance_factor < 1.0:
            raise ValueError(f'Distance factor, that regulates bounds on sigma, must be >= 1.0, got {distance_factor}.')

        self.init_sigma = init_sigma
        self.init_alpha = init_alpha
        self.ker_type = ker_type
        self.ker_pow = ker_pow
        self.pd_metric = pd_metric
        self.pd_metric_kwargs = pd_metric_kwargs
        self.pd_calc_n_jobs = pd_calc_n_jobs
        self.distance_factor = distance_factor

    def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray
    ):
        """
        Calculates train-train pairwise distances between elements and similarity matrix;
        assigns dummy internal parameters to the method;
        returns initial (dummy) fitted predictor.

        :param X_train:
        :param y_train:
        :return:
        """
        self.X_train_ = X_train
        self.y_train_ = y_train.reshape(-1, 1) if y_train.ndim == 1 else y_train

        self.X_train_train_dist_ = pairwise_distances(
            X_train, Y=X_train,
            metric=self.pd_metric,
            n_jobs=self.pd_calc_n_jobs,
            **self.validated_metric_kwargs
        )

        self.min_dist_, self.max_dist_ = self._sigma_bounds(
            self.X_train_train_dist_,
            factor=self.distance_factor
        )
        logger.info(f'train-to-train (min, max) distances: ({self.min_dist_}, {self.max_dist_})')
        self.sigma_ = (self.min_dist_ + self.max_dist_) / 2 if self.init_sigma is None else self.init_sigma
        self.alpha_ = self.init_alpha

        logger.info(
            f'fitting in progress with '
            f'initial sigma (kernel width) = {self.sigma_} and '
            f'initial alpha (regularization param) = {self.alpha_}'
        )

        # constructing initial train-train kernel matrix using dummy sigma:
        K, __ = self._kernel_matrix(
            self.X_train_train_dist_,
            self.sigma_,
            return_derivative=False
        )
        # constructing initial regularized train-train kernel matrix using dummy alpha:
        K_reg = K + self.alpha_ * np.eye(*K.shape, dtype=K.dtype)
        # setting pre-optimization kernel regression weights:
        self.weights_ = self._mul_inv_by_vec(K_reg, self.y_train_)

        return self

    def optimize(
            self,
            X_val: np.ndarray,
            y_val: np.ndarray,
            min_reg: float = 1e-8,
            max_reg: float = 1e1,
            shgo_kwargs: dict = None
    ) -> np.ndarray:
        """
        Calculates train-validation pairwise distances between elements and similarity matrix;
        performs optimization of the internal parameters by minimizing validation target loss;
        returns target predictions for validation data.

        :param X_val:
        :param y_val:
        :param min_reg:
        :param max_reg:
        :param shgo_kwargs: dict of kwargs for the SHGO optimization algorithm.
        :return:
        """
        check_is_fitted(self, ('X_train_', 'weights_', 'sigma_', 'alpha_'))
        self.y_val_ = y_val.reshape(-1, 1) if y_val.ndim == 1 else y_val

        self.X_train_val_dist_ = pairwise_distances(
            self.X_train_, Y=X_val,
            metric=self.pd_metric,
            n_jobs=self.pd_calc_n_jobs,
            **self.validated_metric_kwargs
        )

        tmp_min, tmp_max = self._sigma_bounds(
            self.X_train_val_dist_,
            factor=self.distance_factor
        )
        logger.info(f'train-to-val (min, max) distances: ({tmp_min}, {tmp_max})')
        min_dist = min(self.min_dist_, tmp_min)
        max_dist = max(self.max_dist_, tmp_max)

        logger.info(
            f'optimization in progress with '
            f'min/max bounds on sigma (kernel width): ({min_dist}, {max_dist}); '
            f'min/max bounds on alpha (regularization param): ({min_reg}, {max_reg})'
        )

        shgo_kwargs = shgo_kwargs or {}
        self.sigma_, self.alpha_ = self._perform_optimization(
            [(min_dist, max_dist),
             (min_reg, max_reg)],
            **shgo_kwargs
        )

        # constructing kernel matrices using the optimized sigma:
        K_tt, __ = self._kernel_matrix(
            self.X_train_train_dist_,
            self.sigma_,
            return_derivative=False
        )
        K_tv, __ = self._kernel_matrix(
            self.X_train_val_dist_,
            self.sigma_,
            return_derivative=False
        )
        # constructing regularized train-train kernel matrix using the optimized alpha:
        K_reg = K_tt + self.alpha_ * np.eye(*K_tt.shape, dtype=K_tt.dtype)
        # setting post-optimization kernel regression weights:
        self.weights_ = self._mul_inv_by_vec(K_reg, self.y_train_)

        return K_tv.T @ self.weights_  # TODO remove returning post-optim val predictions, return optimized estimator

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Calculates train-test pairwise distances between elements and similarity matrix;
        returns target predictions for test data.

        :param X_test:
        :return:
        """
        check_is_fitted(self, ('X_train_', 'weights_', 'sigma_', 'alpha_'))

        X_train_test_dist = pairwise_distances(
            self.X_train_, Y=X_test,
            metric=self.pd_metric,
            n_jobs=self.pd_calc_n_jobs,
            **self.validated_metric_kwargs
        )

        K_train_test, __ = self._kernel_matrix(
            X_train_test_dist,
            self.sigma_,
            return_derivative=False
        )

        return K_train_test.T @ self.weights_

    def _kernel_matrix(
            self,
            dist: np.ndarray,
            sigma: float,
            return_derivative: bool = False
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Matrix of pairwise distances -> Kernel matrix

        :param dist:
        :param sigma:
        :param return_derivative:
        :return:
        """
        kernel_func = get_kernel_func(self.ker_type)
        # TODO rewrite temporary workaround below to avoid 'if'
        if self.ker_type == 'gen':
            return kernel_func(
                dist, sigma, power=self.ker_pow,
                return_derivative=return_derivative
            )
        return kernel_func(
            dist, sigma,
            return_derivative=return_derivative
        )

    @staticmethod
    def _sigma_bounds(
            D: np.ndarray,
            factor: float = 1.0
    ) -> tuple[float, float]:
        """
        Extracts min/max values from distance matrix and modifies them using factor value.

        :param D:
        :return:
        """
        dists = D[~np.eye(*D.shape, dtype=bool)]  # extracting non-diag elements

        return dists.min() / factor, dists.max() * factor

    def _calculate_loss_and_grad(self, x0: Sequence[float]) -> tuple[float, np.ndarray]:
        """
        # TODO: add comments on derivations and array dimensions

        :param x0: vector of variables to be optimized
        :return: 1: loss value for given params, 2: array of loss derivatives wrt scale (kernel width) and wrt reg param
        """
        tmp_sigma, tmp_alpha = x0  # expected order in x0: (1) kernel width, (2) reg param

        K_tt, K_der_tt = self._kernel_matrix(
            self.X_train_train_dist_,
            tmp_sigma,
            return_derivative=True
        )
        K_tv, K_der_tv = self._kernel_matrix(
            self.X_train_val_dist_,
            tmp_sigma,
            return_derivative=True
        )

        K_reg = K_tt + tmp_alpha * np.eye(*K_tt.shape, dtype=K_tt.dtype)  # K + αI
        self.weights_ = self._mul_inv_by_vec(K_reg, self.y_train_)

        u_t = K_tv.T
        v = self.y_val_ - u_t @ self.weights_
        v_t = v.T
        tmp = -v_t @ u_t

        g_scale = (
                multi_dot([v_t, K_der_tv.T, self.weights_]) +
                tmp @ self._mul_inv_by_vec(K_reg, K_der_tt @ self.weights_)
        )
        g_reg = tmp @ self._mul_inv_by_vec(K_reg, self.weights_)

        n_samples = v.shape[0]
        factor = -2 / n_samples

        f = (v_t @ v / n_samples).item()
        g = np.array([factor * g_scale.item(), factor * g_reg.item()])

        return f, g

    def _perform_optimization(
            self,
            bounds: Sequence[tuple[float, float]],
            **kwargs
    ) -> Sequence[float]:
        """
        :param bounds: sequence of tuples, [(minval1, maxval1), (minval2, maxval2) ...].
                       For now, the order of bounds is (1) kernel width and (2) reg param.
        :return optimized sigma, optimized alpha

        References:
        1) https://shgo.readthedocs.io/en/latest/docs/README.html
        2) https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html

        The chosen local Quasi-Newton optimizer works well for most of the tasks, but
        for the few of them slight internal parameters tuning might help (see below).
        However, try the default first.

        1) If the algorithm got stuck: increase ftol and/or decrease maxfun and maxiter
        in minimizer_kwargs dict. Why the algorithm might get stuck? Apart from memory
        issues, for example, when we try to find minima on flat landscapes, which is a
        typical problem for Quasi-Newton methods.

        2) If the accuracy is not good enough: decrease ftol and/or increase maxfun and
        maxiter. Also, increasing the number of global optimizer sampling points -
        sampling_pts keyword argument - might help.
        """
        t1 = default_timer()

        res = shgo(
            self._calculate_loss_and_grad,
            bounds,
            iters=kwargs.get('n_iter', 2),
            n=kwargs.get('sampling_pts', 100),
            workers=kwargs.get('n_workers', 1),
            options={  # solver options
                'minimize_every_iter': True,  # TODO default to False? and then increase n_iter and sampling_pts
                'local_iter': False,  # TODO restrict to 10?
                'infty_constraints': True,
                'disp': False
            },
            minimizer_kwargs={
                'method': 'L-BFGS-B',
                'bounds': bounds,
                # If 'jac' is True, fun is assumed to return the gradient along with the objective function:
                # it was not possible with SHGO previously, see https://github.com/scipy/scipy/issues/13547
                'jac': True,
                'options': {  # minimizer options
                    'ftol': kwargs.get('opt_ftol', 1e-5),
                    'maxfun': kwargs.get('opt_maxfun', 250),
                    'maxiter': kwargs.get('opt_maxiter', 50)}
            }
        )

        t2 = default_timer()

        logger.info(res)
        logger.info(
            f'Optimization with {kwargs.get("n_workers", 1)} workers '
            f'terminated successfully in {t2 - t1} sec, '
            f'final variables (sigma, alpha): {res.x}'
        )

        return res.x

    @staticmethod
    def _mul_inv_by_vec(
            matrix: np.ndarray,
            vec: np.ndarray
    ) -> np.ndarray:
        """
        Solves the linear equation set ``a * x = b`` for the unknown ``x``
        for square ``a`` matrix.

        :param matrix: matrix to be inverted
        :param vec: vector to be multiplied by inverted matrix
        """
        try:
            result = linalg.solve(matrix, vec, assume_a='pos', overwrite_a=False)
        except linalg.LinAlgError:
            logger.warning(
                'Eigenvalue computation failed assuming positive semidefiniteness of a constructed matrix. '
                'Applying default "sym" (symmetric) matrix type instead of "pos" (positive-semidefinite).'
            )
            try:
                result = linalg.solve(matrix, vec, assume_a='sym', overwrite_a=False)
            except linalg.LinAlgError:
                logger.warning('Singular matrix in solving dual problem. Using least-squares solution instead.')
                result = linalg.lstsq(matrix, vec)[0]

        return result

    @staticmethod
    def matrix_is_psd(
            matrix: np.ndarray,
            tol: float = 1e-10
    ) -> bool:
        return np.all(linalg.eigvalsh(matrix) >= -tol)

    @property
    def validated_metric_kwargs(self) -> dict:
        """
        # TODO add kwargs validation for all scipy metrics; fit to SciPy cdist kwargs
        """
        if self.pd_metric not in _VALID_METRICS:
            raise ValueError(f'Wrong metric name, please choose from: {_VALID_METRICS}.')

        kwargs = {} if self.pd_metric_kwargs is None else self.pd_metric_kwargs.copy()
        # setting 'p' = 2 (scipy default) for 'minkowski' only:
        if self.pd_metric == 'minkowski':
            kwargs.setdefault('p', 2.0)  # setdefault does not change existing key's value
        # otherwise, we simply don't need 'p'
        elif 'p' in kwargs:
            kwargs.pop('p', None)

        return kwargs

    @property
    def kernel_width(self) -> float:
        return self.sigma_

    @property
    def regularization_parameter(self) -> float:
        return self.alpha_

    @property
    def regression_weights(self) -> np.ndarray:
        return self.weights_
