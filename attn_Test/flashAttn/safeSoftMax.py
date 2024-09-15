from typing import List

import numpy as np


class SoftMaxWithTiling(object):
    def forward(self, x: List[float]):
        # loop 1: get the maximum value of x and the accumulated exponential values
        max_x = -np.inf
        accum_exp = 0.
        for t in x:
            max_x_new = t if t > max_x else max_x
            accum_exp = np.exp(max_x - max_x_new) * accum_exp + np.exp(t - max_x_new)
            max_x = max_x_new

        # loop 2: get the softmax output by dividing the exponential of `x-max(x)` with `accum_exp`
        out = [0. for _ in range(len(x))]
        for i, t in enumerate(x):
            out[i] = np.exp(t - max_x) / accum_exp

        return out


    