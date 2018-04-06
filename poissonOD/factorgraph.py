# -*- coding: utf-8 -*-
"""
   trueskill.factorgraph
   ~~~~~~~~~~~~~~~~~~~~~
   This module contains nodes for the factor graph of TrueSkill algorithm.
   :copyright: (c) 2012-2016 by Heungsub Lee.
   :license: BSD, see LICENSE for more details.
"""
from __future__ import absolute_import
import math
from trueskill.factorgraph import Factor

__all__ = ['TruncateFactorVariational']


class TruncateFactorVariational(Factor):
    """
    TruncateFactor
    Here's an example of the PoissonOD graph factor match::
              rating_layer:  O   O  (PriorFactor)
                             |   |
                             |   |
                perf_layer:  O   O  (LikelihoodFactor)
                             |   |
                             |   |
           team_perf_layer:  O   O  (SumFactor) <- Present on TrueSkill but ignore in PoissonOD
                              \ /
                               |
           team_diff_layer:    O    (SumFactor)
                               |
                               |
               trunc_layer:    O    (TruncateFactor)

    """

    def __init__(self, var, score):
        super(TruncateFactorVariational, self).__init__([var])
        self.k = 0
        self.score = score

    def up(self):
        val = self.var
        msg = self.var[self]
        div = val / msg

        # 'div.pi' denotes the 'precision', which is defined as 1/sigma^2, 'div.tau' denotes
        # the 'precision adjusted mean', which is defined as mu/sigma^2. Naturally, the variance
        # is the inverse of div.pi and the mean is div.tau*variance.

        sig = (1.0/div.pi)     # variance
        mu = div.tau * sig   # mean
        for i in range(5):
            numerator = mu + (self.score*sig) - 1 - self.k + math.sqrt((self.k - mu - (self.score*sig) - 1)**2 + 2*sig)
            denominator = 2.0*sig
            self.k = math.log(numerator/denominator)

        new_mu = sig * (self.score - math.e**self.k) + mu
        new_sig = sig / (1 + sig*(math.e**self.k))

        pi, tau = (1/new_sig), (new_mu/new_sig)
        return val.update_value(self, pi, tau)