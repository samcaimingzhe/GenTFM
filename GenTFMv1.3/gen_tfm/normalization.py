"""Shared float32 min-max transforms; constant columns map to zero."""

import numpy as np


def minmax_stats(values):
    x = np.asarray(values, dtype=np.float32)
    minimum = x.min(axis=0, keepdims=True)
    maximum = x.max(axis=0, keepdims=True)
    return minimum, maximum


def minmax_normalize(values, minimum, maximum):
    x = np.asarray(values, dtype=np.float32)
    minimum = np.asarray(minimum, dtype=np.float32)
    span = np.asarray(maximum, dtype=np.float32) - minimum
    scale = np.where(span > 0, span, np.float32(1))
    return np.asarray((x - minimum) / scale, dtype=np.float32)


def minmax_denormalize(values, minimum, maximum):
    minimum = np.asarray(minimum, dtype=np.float32)
    span = np.asarray(maximum, dtype=np.float32) - minimum
    return np.asarray(np.asarray(values, dtype=np.float32) * span + minimum, dtype=np.float32)
