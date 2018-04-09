# -*- coding: utf-8 -*-
import numpy as np

"""Generates partial ordering simulated data.
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"""

__author__ = "Ewerton Oliveira"
__copyright__ = "Copyright 2018"
__credits__ = ["Ewerton Lopes"]
__license__ = "MIT"
__version__ = "0.0.1"
__maintainer__ = "Ewerton Oliveira"
__author_email__ = "ewerton.lopes@polimi.it"
__status__ = "Production"


def generate_data(num_players=6, verbose=True):
    """
    Assume the the following partial order on players where higher up is better:
         0
        /  \
        1   2
         \/
         3
        /  \
       4    5
    We will sample data from this graph, where we let each player beat its children K times.
    This procedure is based on Kevin Murphy's book simulated data.
    For each match, we also sample the score uniformly at random, where the score for the
    loser is always smaller than the winner's.

    In general, we return a matrix with columns labels given by: playerID1, playerID2,
    playerID1-Score, playerID2-Score.
    """

    def children(matrix, parent, num_players):
        """ Return a binary array where parents are flagged with 1"""
        return [i for i in range(num_players) if matrix[parent][i] == 1.]

    # G represents the transition matrix
    # G(i,1) = id of winner for game i (always the first column on G)
    # G(i,2) = id of loser for game i (always the second column on G)
    G = np.zeros((num_players, num_players))
    G[0, [1, 2]] = 1
    G[1, 3] = 1
    G[2, 3] = 1
    G[3, [4, 5]] = 1

    if verbose:
        print "Transition matrix:\n{}".format(G)

    data = []
    for i in range(num_players):
        ch = children(G, i, num_players)
        for j in ch:
            # Sample the number of games between this pair
            K = np.random.choice(range(5), p=[0.1, 0.2, 0.2, 0.1, 0.4])
            for k in range(K):
                win_scr = np.random.randint(1, 11, 1)
                data.append([i, j, win_scr, np.random.randint(0, win_scr, 1)])
    data = np.array(data)

    num_games = []
    for i in range(num_players):
        s = 0
        for j in data:
            if j[0] == i or j[1] == i:
                s += 1
        num_games.append(s)

    if verbose:
        print " #of games p/ player: {}".format(num_games)

    return data, num_games


if __name__ == '__main__':
    #d, ng = generate_data()
    #np.savetxt("par_order.csv", d, fmt='%d', delimiter=",")
    import pandas as pd

    data = pd.read_csv('murphy.csv', header=None).as_matrix()
    for l in data:
        l[2] = win_scr = np.random.randint(1, 11, 1)
        l[3] = np.random.randint(0, win_scr, 1)
    print data

    np.savetxt("murphy_score.csv", data, fmt='%d', delimiter=",")