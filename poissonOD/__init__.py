#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""Score-based"""
from __future__ import absolute_import

from itertools import chain
import math

from six import iteritems
from six.moves import map, range, zip

from .__about__ import __version__
from .backends import choose_backend
from .factorgraph import (LikelihoodFactor, PriorFactor, SumFactor,
                          TruncateFactorVariational, Variable)
from .mathematics import Gaussian, Matrix

__all__ = [
    # PoissonOD objects
    'PoissonOD', 'Rating',
    # functions for the global environment
    'rate', 'quality', 'rate_1vs1', 'quality_1vs1', 'expose', 'setup',
    'global_env',
    # default values
    'MU', 'SIGMA', 'BETA', 'TAU', 'DRAW_PROBABILITY',
    # draw probability helpers
    # 'calc_draw_probability', 'calc_draw_margin',
]

#: Default initial mean of ratings.
MU = 25.
#: Default initial standard deviation of ratings.
SIGMA = MU / 3
#: Default distance that guarantees about 76% chance of winning.
BETA = SIGMA / 2
#: Default dynamic factor.
TAU = SIGMA / 100
#: Default draw probability of the game.
DRAW_PROBABILITY = .10
#: A basis to check reliability of the result.
DELTA = 0.0001

def setup(mu=MU, sigma=SIGMA, beta=BETA, tau=TAU,
          draw_probability=DRAW_PROBABILITY, backend=None, env=None):
    """Setups the global environment.
    :param env: the specific :class:`TrueSkill` object to be the global
                environment.  It is optional.
    >>> Rating()
    trueskill.Rating(mu=25.000, sigma=8.333)
    >>> setup(mu=50)  #doctest: +ELLIPSIS
    trueskill.TrueSkill(mu=50.000, ...)
    >>> Rating()
    trueskill.Rating(mu=50.000, sigma=8.333)
    """
    if env is None:
        env = PoissonOD(mu, sigma, beta, tau, draw_probability, backend)
    global_env.__trueskill__ = env
    return env


def global_env():
    """Gets the :class:`TrueSkill` object which is the global environment."""
    try:
        global_env.__trueskill__
    except AttributeError:
        # setup the default environment
        setup()
    return global_env.__trueskill__


def rate(rating_groups, ranks=None, weights=None, min_delta=DELTA):
    """A proxy function for :meth:`TrueSkill.rate` of the global environment.
    .. versionadded:: 0.2
    """
    return global_env().rate(rating_groups, ranks, weights, min_delta)


def quality(rating_groups, weights=None):
    """A proxy function for :meth:`TrueSkill.quality` of the global
    environment.
    .. versionadded:: 0.2
    """
    return global_env().quality(rating_groups, weights)


def quality_1vs1(rating1, rating2, env=None):
    """A shortcut to calculate the match quality between just 2 players in
    a head-to-head match::
       if quality_1vs1(alice, bob) < 0.50:
           print('This match seems to be not so fair')
    :param rating1: the rating.
    :param rating2: the another rating.
    :param env: the :class:`TrueSkill` object.  Defaults to the global
                environment.
    .. versionadded:: 0.2
    """
    if env is None:
        env = global_env()
    return env.quality([(rating1,), (rating2,)])


def expose(rating):
    """A proxy function for :meth:`TrueSkill.expose` of the global environment.
    .. versionadded:: 0.4
    """
    return global_env().expose(rating)


def _team_sizes(rating_groups):
    """Makes a size map of each teams."""
    team_sizes = [0]
    for group in rating_groups:
        team_sizes.append(len(group) + team_sizes[-1])
    del team_sizes[0]
    return team_sizes


class Rating(Gaussian):
    """Represents a player's skill as Gaussian distrubution.
    The default mu and sigma value follows the global environment's settings.
    If you don't want to use the global, use :meth:`TrueSkill.create_rating` to
    create the rating object.
    :param mu: the mean.
    :param sigma: the standard deviation.
    """

    def __init__(self, mu=None, sigma=None):
        if isinstance(mu, tuple):
            mu, sigma = mu
        elif isinstance(mu, Gaussian):
            mu, sigma = mu.mu, mu.sigma
        if mu is None:
            mu = global_env().mu
        if sigma is None:
            sigma = global_env().sigma
        super(Rating, self).__init__(mu, sigma)

    def __int__(self):
        return int(self.mu)

    def __long__(self):
        return long(self.mu)

    def __float__(self):
        return float(self.mu)

    def __iter__(self):
        return iter((self.mu, self.sigma))

    def __repr__(self):
        c = type(self)
        args = ('.'.join([c.__module__, c.__name__]), self.mu, self.sigma)
        return '%s(mu=%.3f, sigma=%.3f)' % args


class PoissonOD:

    def __init__(self, mu=MU, sigma=SIGMA, beta=BETA, tau=TAU,
                 draw_probability=DRAW_PROBABILITY, backend=None):
        self.mu = mu
        self.sigma = sigma
        self.beta = beta
        self.tau = tau
        self.draw_probability = draw_probability
        self.backend = backend
        if isinstance(backend, tuple):
            self.cdf, self.pdf, self.ppf = backend
        else:
            self.cdf, self.pdf, self.ppf = choose_backend(backend)

    def validate_rating_groups(self, rating_groups):
        """Validates a ``rating_groups`` argument.  It should contain more than
        2 groups and all groups must not be empty.
        >>> env = TrueSkill()
        >>> env.validate_rating_groups([])
        Traceback (most recent call last):
            ...
        ValueError: need multiple rating groups
        >>> env.validate_rating_groups([(Rating(),)])
        Traceback (most recent call last):
            ...
        ValueError: need multiple rating groups
        >>> env.validate_rating_groups([(Rating(),), ()])
        Traceback (most recent call last):
            ...
        ValueError: each group must contain multiple ratings
        >>> env.validate_rating_groups([(Rating(),), (Rating(),)])
        ... #doctest: +ELLIPSIS
        [(truekill.Rating(...),), (trueskill.Rating(...),)]
        """
        # check group sizes
        if len(rating_groups) < 2:
            raise ValueError('Need multiple rating groups')
        elif not all(rating_groups):
            raise ValueError('Each group must contain multiple ratings')
        # check group types
        group_types = set(map(type, rating_groups))
        if len(group_types) != 1:
            raise TypeError('All groups should be same type')
        elif group_types.pop() is Rating:
            raise TypeError('Rating cannot be a rating group')
        # normalize rating_groups
        if isinstance(rating_groups[0], dict):
            dict_rating_groups = rating_groups
            rating_groups = []
            keys = []
            for dict_rating_group in dict_rating_groups:
                rating_group, key_group = [], []
                for key, rating in iteritems(dict_rating_group):
                    rating_group.append(rating)
                    key_group.append(key)
                rating_groups.append(tuple(rating_group))
                keys.append(tuple(key_group))
        else:
            rating_groups = list(rating_groups)
            keys = None
        return rating_groups, keys

    def validate_weights(self, weights, rating_groups, keys=None):
        if weights is None:
            weights = [(1,) * len(g) for g in rating_groups]
        elif isinstance(weights, dict):
            weights_dict, weights = weights, []
            for x, group in enumerate(rating_groups):
                w = []
                weights.append(w)
                for y, rating in enumerate(group):
                    if keys is not None:
                        y = keys[x][y]
                    w.append(weights_dict.get((x, y), 1))
        return weights

    def factor_graph_builders(self, rating_groups, ranks, weights, score):
        """Makes nodes for the PoissonOD factor graph.
        Here's an example of a TrueSkill factor graph when 1 vs 2 vs 1 match::
              rating_layer:  O O O O  (PriorFactor)
                             | | | |
                             | | | |
                perf_layer:  O O O O  (LikelihoodFactor)
                             | \ / |
                             |  |  |
           team_perf_layer:  O  O  O  (SumFactor)
                             \ / \ /
                              |   |
           team_diff_layer:   O   O   (SumFactor)
                              |   |
                              |   |
               trunc_layer:   O   O   (TruncateFactor)
        """
        flatten_ratings = sum(map(tuple, rating_groups), ())
        flatten_weights = sum(map(tuple, weights), ())
        size = len(flatten_ratings)
        group_size = len(rating_groups)
        # create variables
        rating_vars = [Variable() for x in range(size)]
        perf_vars = [Variable() for x in range(size)]
        team_perf_vars = [Variable() for x in range(group_size)]
        team_diff_vars = [Variable() for x in range(group_size - 1)]
        team_sizes = _team_sizes(rating_groups)

        # layer builders
        def build_rating_layer():
            for rating_var, rating in zip(rating_vars, flatten_ratings):
                yield PriorFactor(rating_var, rating, self.tau)

        def build_perf_layer():
            for rating_var, perf_var in zip(rating_vars, perf_vars):
                yield LikelihoodFactor(rating_var, perf_var, self.beta ** 2)

        def build_team_perf_layer():
            for team, team_perf_var in enumerate(team_perf_vars):
                if team > 0:
                    start = team_sizes[team - 1]
                else:
                    start = 0
                end = team_sizes[team]
                child_perf_vars = perf_vars[start:end]
                coeffs = flatten_weights[start:end]
                yield SumFactor(team_perf_var, child_perf_vars, coeffs)

        def build_team_diff_layer():
            for team, team_diff_var in enumerate(team_diff_vars):
                yield SumFactor(team_diff_var,
                                team_perf_vars[team:team + 2], [+1, -1])

        def build_trunc_layer():
            for x, team_diff_var in enumerate(team_diff_vars):
                yield TruncateFactorVariational(team_diff_var, score)

        # build layers
        return (build_rating_layer, build_perf_layer, build_team_perf_layer,
                build_team_diff_layer, build_trunc_layer)

    def run_schedule(self, build_rating_layer, build_perf_layer,
                     build_team_perf_layer, build_team_diff_layer,
                     build_trunc_layer, min_delta=DELTA):
        """Sends messages within every nodes of the factor graph until the
        result is reliable.
        """
        if min_delta <= 0:
            raise ValueError('min_delta must be greater than 0')
        layers = []

        def build(builders):
            layers_built = [list(build()) for build in builders]
            layers.extend(layers_built)
            return layers_built

        # gray arrows
        layers_built = build([build_rating_layer,
                              build_perf_layer,
                              build_team_perf_layer])
        rating_layer, perf_layer, team_perf_layer = layers_built
        for f in chain(*layers_built):
            f.down()
        # arrow #1, #2, #3
        team_diff_layer, trunc_layer = build([build_team_diff_layer,
                                              build_trunc_layer])
        team_diff_len = len(team_diff_layer)
        for x in range(10):
            if team_diff_len == 1:
                # only two teams
                team_diff_layer[0].down()
                delta = trunc_layer[0].up()
            else:
                # multiple teams
                delta = 0
                for x in range(team_diff_len - 1):
                    team_diff_layer[x].down()
                    delta = max(delta, trunc_layer[x].up())
                    team_diff_layer[x].up(1)  # up to right variable
                for x in range(team_diff_len - 1, 0, -1):
                    team_diff_layer[x].down()
                    delta = max(delta, trunc_layer[x].up())
                    team_diff_layer[x].up(0)  # up to left variable
            # repeat until to small update
            if delta <= min_delta:
                break

        # up both ends
        team_diff_layer[0].up(0)
        team_diff_layer[team_diff_len - 1].up(1)
        # up the remainder of the black arrows
        for f in team_perf_layer:
            for x in range(len(f.vars) - 1):
                f.up(x)
        for f in perf_layer:
            f.up()
        return layers

    def rate(self, rating_groups, score, ranks=None, weights=None, min_delta=DELTA):
        """Recalculates ratings by the ranking table::env = TrueSkill()  # uses default settings
           # create ratings
           r1 = env.create_rating(42.222)
           r2 = env.create_rating(89.999)
           # calculate new ratings
           rating_groups = [(r1,), (r2,)]
           rated_rating_groups = env.rate(rating_groups, ranks=[0, 1])
           # save new ratings
           (r1,), (r2,) = rated_rating_groups
        ``rating_groups`` is a list of rating tuples or dictionaries that
        represents each team of the match.  You will get a result as same
        structure as this argument.  Rating dictionaries for this may be useful
        to choose specific player's new rating::
           # load players from the database
           p1 = load_player_from_database('Arpad Emrick Elo')
           p2 = load_player_from_database('Mark Glickman')
           p3 = load_player_from_database('Heungsub Lee')
           # calculate new ratings
           rating_groups = [{p1: p1.rating, p2: p2.rating}, {p3: p3.rating}]
           rated_rating_groups = env.rate(rating_groups, ranks=[0, 1])
           # save new ratings
           for player in [p1, p2, p3]:
               player.rating = rated_rating_groups[player.team][player]
        :param rating_groups: a list of tuples or dictionaries containing
                              :class:`Rating` objects.
        :param ranks: a ranking table.  By default, it is same as the order of
                      the ``rating_groups``.
        :param weights: weights of each players for "partial play".
        :param min_delta: each loop checks a delta of changes and the loop
                          will stop if the delta is less then this argument.
        :returns: recalculated ratings same structure as ``rating_groups``.
        :raises: :exc:`FloatingPointError` occurs when winners have too lower
                 rating than losers.  higher floating-point precision couls
                 solve this error.  set the backend to "mpmath".
        .. versionadded:: 0.2
        """

        rating_groups, keys = self.validate_rating_groups(rating_groups)
        weights = self.validate_weights(weights, rating_groups, keys)
        group_size = len(rating_groups)
        if ranks is None:
            ranks = range(group_size)
        elif len(ranks) != group_size:
            raise ValueError('Wrong ranks')
        # sort rating groups by rank
        by_rank = lambda x: x[1][1]
        sorting = sorted(enumerate(zip(rating_groups, ranks, weights)),
                         key=by_rank)
        sorted_rating_groups, sorted_ranks, sorted_weights = [], [], []
        for x, (g, r, w) in sorting:
            sorted_rating_groups.append(g)
            sorted_ranks.append(r)
            # make weights to be greater than 0
            sorted_weights.append(max(min_delta, w_) for w_ in w)
        # build factor graph
        args = (sorted_rating_groups, sorted_ranks, sorted_weights, score)
        builders = self.factor_graph_builders(*args)
        args = builders + (min_delta,)
        layers = self.run_schedule(*args)
        # make result
        rating_layer, team_sizes = layers[0], _team_sizes(sorted_rating_groups)
        transformed_groups = []
        for start, end in zip([0] + team_sizes[:-1], team_sizes):
            group = []
            for f in rating_layer[start:end]:
                group.append(Rating(float(f.var.mu), float(f.var.sigma)))
            transformed_groups.append(tuple(group))
        by_hint = lambda x: x[0]
        unsorting = sorted(zip((x for x, __ in sorting), transformed_groups),
                           key=by_hint)
        if keys is None:
            return [g for x, g in unsorting]
        # restore the structure with input dictionary keys
        return [dict(zip(keys[x], g)) for x, g in unsorting]


def rate_1vs1(rating1, rating2, score, drawn=False, min_delta=DELTA, env=None):
    """A shortcut to rate just 2 players in a head-to-head match::
       alice, bob = Rating(25), Rating(30)
       alice, bob = rate_1vs1(alice, bob)
       alice, bob = rate_1vs1(alice, bob, drawn=True)
    :param rating1: the winner's rating if they didn't draw.
    :param rating2: the loser's rating if they didn't draw.
    :param drawn: if the players drew, set this to ``True``.  Defaults to
                  ``False``.
    :param min_delta: will be passed to :meth:`rate`.
    :param env: the :class:`TrueSkill` object.  Defaults to the global
                environment.
    :returns: a tuple containing recalculated 2 ratings.
    .. versionadded:: 0.2
    """
    if env is None:
        env = global_env()
    ranks = [0, 0 if drawn else 1]
    teams = env.rate([(rating1,), (rating2,)], score, ranks, min_delta=min_delta)
    return teams[0][0], teams[1][0]