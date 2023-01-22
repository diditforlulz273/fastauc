import ctypes
import os
from typing import Union
import numpy as np
from numpy.ctypeslib import ndpointer
import numba


class CppAuc:
    """A python wrapper class for a C++ library, used to load it once and make fast calls after.
    NB be aware of data types accepted, see method docstrings. 
    """

    def __init__(self):
        self._handle = ctypes.CDLL(os.path.dirname(os.path.realpath(__file__)) + "/cpp_auc.so")
        self._handle.cpp_auc_ext.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                             ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
                                             ctypes.c_size_t,
                                             ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                             ctypes.c_size_t
                                             ]
        self._handle.cpp_auc_ext.restype = ctypes.c_float

    def roc_auc_score(self, y_true: np.array, y_score: np.array, sample_weight: np.array=None) -> float:
        """a method to calculate AUC via C++ lib.

        Args:
            y_true (np.array): 1D numpy array of dtype=np.bool8 as true labels.
            y_score (np.array): 1D numpy array of dtype=np.float32 as probability predictions.
            sample_weight (np.array): 1D numpy array as sample weights, optional.

        Returns:
            float: AUC score
        """
        n = len(y_true)
        n_sample_weights = len(sample_weight) if sample_weight is not None else 0
        sample_weight = sample_weight if sample_weight is not None else np.array([],dtype=np.float32)
        result = self._handle.cpp_auc_ext(y_score, y_true, n, sample_weight, n_sample_weights)
        return result

    def roc_auc_score_batch(self, y_true: np.array, y_score: np.array) -> np.array:
        raise NotImplemented
        return np.array(result)

def fast_numba_auc(y_true: np.array, y_score: np.array, sample_weight: np.array=None) -> float:
    """a function to calculate AUC via python + numba.

    Args:
        y_true (np.array): 1D numpy array as true labels.
        y_score (np.array): 1D numpy array as probability predictions.
        sample_weight (np.array): 1D numpy array as sample weights, optional.

    Returns:
        AUC score as float
    """
    if sample_weight is None:
        return fast_numba_auc_nonw(y_true=y_true, y_score=y_score)
    else:
        return fast_numba_auc_w(y_true=y_true, y_score=y_score, sample_weight=sample_weight)


@numba.njit
def trapezoid_area(x1: float, x2: float, y1: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    return dx * y1 + dy * dx / 2.0


@numba.njit
def fast_numba_auc_nonw(y_true: np.array, y_score: np.array) -> float:
    y_true = (y_true == 1)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    prev_fps = 0
    prev_tps = 0
    last_counted_fps = 0
    last_counted_tps = 0
    auc = 0.0
    for i in range(len(y_true)):
        tps = prev_tps + y_true[i]
        fps = prev_fps + (1 - y_true[i])
        if i == len(y_true) - 1 or y_score[i+1] != y_score[i]:
            auc += trapezoid_area(last_counted_fps, fps, last_counted_tps, tps)
            last_counted_fps = fps
            last_counted_tps = tps
        prev_tps = tps
        prev_fps = fps
    return auc / (prev_tps*prev_fps)

@numba.njit
def fast_numba_auc_w(y_true: np.array, y_score: np.array, sample_weight: np.array) -> float:
    y_true = (y_true == 1)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    sample_weight = sample_weight[desc_score_indices]

    prev_fps = 0
    prev_tps = 0
    last_counted_fps = 0
    last_counted_tps = 0
    auc = 0.0
    for i in range(len(y_true)):
        weight = sample_weight[i]
        tps = prev_tps + y_true[i] * weight
        fps = prev_fps + (1 - y_true[i]) * weight
        if i == len(y_true) - 1 or y_score[i+1] != y_score[i]:
            auc += trapezoid_area(last_counted_fps, fps, last_counted_tps, tps)
            last_counted_fps = fps
            last_counted_tps = tps
        prev_tps = tps
        prev_fps = fps
    return auc / (prev_tps * prev_fps)


def fast_auc(y_true: np.array, y_score: np.array, sample_weight: np.array=None) -> Union[float, str]:
    """a function to calculate AUC via python.

    Args:
        y_true (np.array): 1D numpy array as true labels.
        y_score (np.array): 1D numpy array as probability predictions.
        sample_weight (np.array): 1D numpy array as sample weights, optional.

    Returns:
        float or str: AUC score or 'error' if imposiible to calculate
    """
    # binary clf curve
    y_true = (y_true == 1)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        sample_weight = sample_weight[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    if sample_weight is not None:
        tps = np.cumsum(y_true * sample_weight)[threshold_idxs]
        fps = np.cumsum((1 - y_true) * sample_weight)[threshold_idxs]
    else:
        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps

    # roc
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if fps[-1] <= 0 or tps[-1] <= 0:
        return np.nan

    # auc
    direction = 1
    dx = np.diff(fps)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            return 'error'

    area = direction * np.trapz(tps, fps) / (tps[-1] * fps[-1])

    return area
