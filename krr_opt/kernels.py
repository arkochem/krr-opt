# -*- coding: utf-8 -*-

import numpy as np

from typing import Optional

__all__ = [
    'get_kernel_func',
    'VALID_KERNELS'
]

VALID_KERNELS = (
    "gau",
    "lap",
    "mat52",
    "gen"
)


def get_kernel_func(func_name):
    return {
        'gau': _gaussian,
        'lap': _laplacian,
        'mat52': _matern52,
        'gen': _generic
    }[func_name]


def _gaussian(
        dist: np.ndarray,
        sigma: float,
        return_derivative: bool = False
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    # TODO description + implementation of derivatives
    # TODO should be returned by _generic with power=2.0

    :param dist:
    :param sigma:
    :param return_derivative:
    :return:
    """
    dist_sq = np.square(dist)
    K = np.exp(-dist_sq / (2 * (sigma ** 2)))  # dist_sq - squared already

    if not return_derivative:
        return np.nan_to_num(K), None

    derivative = dist_sq * K / (sigma ** 3)
    return np.nan_to_num(K), np.nan_to_num(derivative)


def _laplacian(
        dist: np.ndarray,
        sigma: float,
        return_derivative: bool = False
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    # TODO description + implementation of derivatives
    # TODO should be returned by _generic with power=1.0

    :param dist:
    :param sigma:
    :param return_derivative:
    :return:
    """
    K = np.exp(-dist / sigma)  # dist - not squared

    if not return_derivative:
        return np.nan_to_num(K), None

    derivative = dist * K / (sigma ** 2)
    return np.nan_to_num(K), np.nan_to_num(derivative)


def _matern52(
        dist: np.ndarray,
        sigma: float,
        return_derivative: bool = False
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    # TODO description + implementation of derivatives

    :param dist:
    :param sigma:
    :param return_derivative:
    :return:
    """
    K1 = dist * np.sqrt(5) / sigma
    K1_sq = K1 ** 2
    K1_poly = (1.0 + K1 + K1_sq / 3.0)
    K1_exp = np.exp(-K1)
    K = K1_poly * K1_exp

    if not return_derivative:
        return np.nan_to_num(K), None

    tmp = -K1 / sigma
    derivative = K1_exp * (tmp + -2.0 * K1_sq / (3.0 * sigma)) - K * tmp
    return np.nan_to_num(K), np.nan_to_num(derivative)


def _generic(
        dist: np.ndarray,
        sigma: float,
        power: float = 1.0,
        return_derivative: bool = False
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    # TODO description + implementation of derivatives

    :param dist:
    :param sigma:
    :param power:
    :param return_derivative:
    :return:
    """
    dist2pow = dist ** power
    K = np.exp(-dist2pow / (sigma ** power))

    if not return_derivative:
        return np.nan_to_num(K), None

    derivative = power * dist2pow * K / (sigma ** (power + 1))
    return np.nan_to_num(K), np.nan_to_num(derivative)
