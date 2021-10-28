import ctypes
import os
from typing import Union
import numpy as np
from numpy.ctypeslib import ndpointer


class CppAuc:
    """A python wrapper class for a C++ library, used to load it once and make fast calls after.
    NB be aware of data types accepted, see method docstrings. 
    """

    def __init__(self):
        self._handle = ctypes.CDLL(os.path.dirname(os.path.realpath(__file__)) + "/cpp_auc.so")
        self._handle.cpp_auc_ext.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
                                             ndpointer(ctypes.c_bool, flags="C_CONTIGUOUS"),
                                             ctypes.c_size_t
                                             ]
        self._handle.cpp_auc_ext.restype = ctypes.c_float

    def roc_auc_score(self, y_true: np.array, y_score: np.array) -> float:
        """a method to calculate AUC via C++ lib.

        Args:
            y_true (np.array): 1D numpy array of dtype=np.bool8 as true labels.
            y_score (np.array): 1D numpy array of dtype=np.float32 as probability predictions.

        Returns:
            float: AUC score
        """
        n = len(y_true)
        result = self._handle.cpp_auc_ext(y_score, y_true, n)
        return result

    def roc_auc_score_batch(self, y_true: np.array, y_score: np.array) -> np.array:
        raise NotImplemented
        return np.array(result)


def fast_auc(y_true: np.array, y_prob: np.array) -> Union[float, str]:
    """a function to calculate AUC via python.

    Args:
        y_true (np.array): 1D numpy array as true labels.
        y_score (np.array): 1D numpy array as probability predictions.

    Returns:
        float or str: AUC score or 'error' if imposiible to calculate
    """
    # binary clf curve
    y_true = (y_true == 1)

    desc_score_indices = np.argsort(y_prob, kind="mergesort")[::-1]
    y_score = y_prob[desc_score_indices]
    y_true = y_true[desc_score_indices]

    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = y_score[threshold_idxs]

    if len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    # roc
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    # auc
    direction = 1
    dx = np.diff(fpr)
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            return 'error'

    area = direction * np.trapz(tpr, fpr)

    return area
