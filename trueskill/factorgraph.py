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

from six.moves import zip

from .mathematics import Gaussian, inf


__all__ = ['Variable', 'PriorFactor', 'LikelihoodFactor', 'SumFactor',
           'TruncateFactor']


class Node(object):
    pass


class Variable(Node, Gaussian):
    """ The factor graph variable nodes."""

    def __init__(self):
        # The dictionary of messages the variable can pass.
        self.messages = {}
        super(Variable, self).__init__()

    def set(self, val):
        """
        Set the variable parameters.
        :param val: An updating gaussian.
        :return: The numerical change in the variable (gaussian) parameters
                 caused by the updating gaussian.
        """
        delta = self.delta(val)
        self.pi  = val.pi       # precision: 1/sigma^2
        self.tau = val.tau      # precision adjusted mean: precision*mean.
        return delta

    def delta(self, other):
        """
        Quantifies the difference between gaussian parameters
        :param other:
        :return: max of the difference in 'tau' (precision adjusted mean)
                 or the sqrt of the difference in precision (pi)
        """
        pi_delta = abs(self.pi - other.pi)
        if pi_delta == inf:
            return 0.
        return max(abs(self.tau - other.tau), math.sqrt(pi_delta))

    def update_message(self, factor, pi=0, tau=0, message=None):
        message = message or Gaussian(pi=pi, tau=tau)
        old_message, self[factor] = self[factor], message
        return self.set(self / old_message * message)

    def update_value(self, factor, pi=0, tau=0, value=None):
        """
        Update variable parameters. This also updates the value of the
        'message-from-factor' for the variable, which corresponds to 'Step 1'
        (Compute marginal skills) and 'Step 2' (Compute skill to game messages).

        :param factor: a factor from which a variable send message to.
        :param pi: a precision to be incorporated.
        :param tau: a precision adjusted mean to be incorporated.
        :param value: A gaussian.
        :return: the numerical difference value from updating.
        """
        value = value or Gaussian(pi=pi, tau=tau)
        old_message = self[factor]                      # get old value of variable 'message-from-factor'.
        self[factor] = value * old_message / self       # update variable message from factor.
        return self.set(value)                          # update gaussian value.

    def __getitem__(self, factor):
        return self.messages[factor]

    def __setitem__(self, factor, message):
        self.messages[factor] = message

    def __repr__(self):
        args = (type(self).__name__, super(Variable, self).__repr__(),
                len(self.messages), '' if len(self.messages) == 1 else 's')
        return '<%s %s with %d connection%s>' % args


class Factor(Node):

    def __init__(self, vars):
        self.vars = vars
        for var in vars:
            var[self] = Gaussian()

    def down(self):
        return 0

    def up(self):
        return 0

    @property
    def var(self):
        assert len(self.vars) == 1
        return self.vars[0]

    def __repr__(self):
        args = (type(self).__name__, len(self.vars),
                '' if len(self.vars) == 1 else 's')
        return '<%s with %d connection%s>' % args


class PriorFactor(Factor):

    def __init__(self, variable, rating, dynamic=0):
        """
        :param variable: a Variable() instance
        :param rating: A player Rating() instance.
        :param dynamic: Without this parameter, the TrueSkill algorithm would always cause the player’s standard
                        deviation term to shrink and therefore become more certain about a player. Before skill
                        updates are calculated, we add in to the player’s skill variance. This ensures that the
                        game retains “dynamics.” That is, the dynamic parameter determines how easy it will be
                        for a player to move up and down a leaderboard. A larger value for dynamic will tend to
                        cause more volatility of player positions.
        """
        super(PriorFactor, self).__init__([variable])
        self.val = rating
        self.dynamic = dynamic

    def down(self):
        """Computes prior down message."""
        sigma = math.sqrt(self.val.sigma ** 2 + self.dynamic ** 2)  # sqrt(sigma^2 + dynamic^2), variance addition.
        value = Gaussian(self.val.mu, sigma)
        return self.var.update_value(self, value=value)


class LikelihoodFactor(Factor):

    def __init__(self, rating_var, perf_var, variance):
        super(LikelihoodFactor, self).__init__([rating_var, perf_var])
        self.rating_var = rating_var
        self.perf_var = perf_var
        self.variance = variance

    def calc_a(self, var):
        return 1. / (1. + self.variance * var.pi)

    def down(self):
        # update perf_var.
        msg = self.rating_var / self.rating_var[self]
        a = self.calc_a(msg)
        return self.perf_var.update_message(self, a * msg.pi, a * msg.tau)

    def up(self):
        # update rating_var.
        msg = self.perf_var / self.perf_var[self]
        a = self.calc_a(msg)
        return self.rating_var.update_message(self, a * msg.pi, a * msg.tau)


class SumFactor(Factor):

    def __init__(self, sum_var, term_vars, coeffs):
        super(SumFactor, self).__init__([sum_var] + term_vars)
        self.sum = sum_var              # plays the role of team diff var.
        self.terms = term_vars          # performance vars
        self.coeffs = coeffs            # [+1, -1]

    def down(self):
        vals = self.terms
        msgs = [var[self] for var in vals]                      # msgs = messages from diff_factor to perf_var
        return self.update(self.sum, vals, msgs, self.coeffs)

    def up(self, index=0):
        coeff = self.coeffs[index]
        coeffs = []
        for x, c in enumerate(self.coeffs):
            try:
                if x == index:
                    coeffs.append(1. / coeff)
                else:
                    coeffs.append(-c / coeff)
            except ZeroDivisionError:
                coeffs.append(0.)
        vals = self.terms[:]
        vals[index] = self.sum
        msgs = [var[self] for var in vals]
        return self.update(self.terms[index], vals, msgs, coeffs)

    def update(self, var, vals, msgs, coeffs):
        pi_inv = 0
        mu = 0
        for val, msg, coeff in zip(vals, msgs, coeffs):
            div = val / msg         # subtraction of pi and tau variables in 3rd (cell) update equation in the paper
            mu += coeff * div.mu    # this performs the subtraction of the means for the 1 vs 1 case.
            if pi_inv == inf:
                continue
            try:
                # numpy.float64 handles floating-point error by different way.
                # For example, it can just warn RuntimeWarning on n/0 problem
                # instead of throwing ZeroDivisionError.  So div.pi, the
                # denominator has to be a built-in float.
                pi_inv += coeff ** 2 / float(div.pi)
            except ZeroDivisionError:
                pi_inv = inf
        pi = 1. / pi_inv                            # 1st equation in the 3rd cell of the trueskill paper
        tau = pi * mu                               # 2nd equation in the 3rd cell of the trueskill paper
        return var.update_message(self, pi, tau)


class TruncateFactor(Factor):
    """
    Implements the comparison factor. Uses moment matching.
    """

    def __init__(self, var, v_func, w_func, draw_margin):
        super(TruncateFactor, self).__init__([var])
        self.v_func = v_func
        self.w_func = w_func
        self.draw_margin = draw_margin

    def up(self):
        val = self.var                              # team_diff variable
        msg = self.var[self]                        # factor gaussian message
        div = val / msg                             # this division gives you 'c' and 'd' in the TrueSkill paper.
        sqrt_pi = math.sqrt(div.pi)                 # sqrt(div.pi) = 1/sigma. This is sqrt(c) in the TrueSkill paper.
        args = (div.tau / sqrt_pi, self.draw_margin * sqrt_pi)  # args of Wf and Vf in the TrueSkill paper.
        v = self.v_func(*args)                      # calculate v
        w = self.w_func(*args)                      # calculate w
        denom = (1. - w)
        pi, tau = div.pi / denom, (div.tau + sqrt_pi * v) / denom
        return val.update_value(self, pi, tau)      # update the team_diff variable with new values of pi and tau.


class PoissonTruncateFactor(Factor):

    def __init__(self, var, v_func, w_func, draw_margin):
        super(PoissonTruncateFactor, self).__init__([var])
        self.v_func = v_func
        self.w_func = w_func
        self.draw_margin = draw_margin

    def up(self):
        val = self.var
        msg = self.var[self]
        div = val / msg
        sqrt_pi = math.sqrt(div.pi)                                 # this is sigma
        args = (div.tau / sqrt_pi, self.draw_margin * sqrt_pi)
        v = self.v_func(*args)
        w = self.w_func(*args)
        denom = (1. - w)
        pi, tau = div.pi / denom, (div.tau + sqrt_pi * v) / denom
        return val.update_value(self, pi, tau)
