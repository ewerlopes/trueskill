import trueskill
from matplotlib import pyplot as plt

# Head-to-head (1 vs. 1) match rule

# Most competition games follows 1:1 match rule. If your game does,
# just use _1vs1 shortcuts containing rate_1vs1() and quality_1vs1().
# These are very easy to use.

# First of all, we need 2 Rating objects:

r1 = trueskill.Rating()  # 1P's skill
r2 = trueskill.Rating()  # 2P's skill

# Then we can guess match quality which is equivalent with draw probability of
# this match using quality_1vs1():

print('{:.1%} chance to draw'.format(trueskill.quality_1vs1(r1, r2)))

# After the game, TrueSkill recalculates their ratings by the game result.
# For example, if 1P beat 2P:

new_r1, new_r2 = trueskill.rate_1vs1(r1, r2)
print(new_r1)
print(new_r2)

# Mu value follows player's win/draw/lose records. Higher value means higher
# game skill. And sigma value follows the number of games. Lower value means
# many game plays and higher rating confidence.

# So 1P, a winner's skill grew up from 25 to 29.396 but 2P, a loser's skill
# shrank to 20.604. And both sigma values became narrow about same magnitude.

# Of course, you can also handle a tie game with drawn=True:
new_r1, new_r2 = trueskill.rate_1vs1(r1, r2, drawn=True)
print(new_r1)
print(new_r2)


# Demo of the TrueSkill model
# PMTKauthor Carl Rasmussen and  Joaquin Quinonero-Candela,
# PMTKurl http://mlg.eng.cam.ac.uk/teaching/4f13/1112
# PMTKmodified Kevin Murphy

import numpy as np
from scipy.stats import norm

N_PLAYERS = 6
np.random.seed(0)


def children(matrix, parent):
    return [i for i in range(N_PLAYERS) if matrix[parent][i] == 1.]


def trueskill_demo():
    """
    Let us assume the following partial order on players
    where higher up is better
          0
        /  \
        1   2
         \/
         3
        /  \
       4    5
    We will sample data from this graph, where we let each player
    beat its children K times"""

    PLAYERS = {
        0: trueskill.Rating(),
        1: trueskill.Rating(),
        2: trueskill.Rating(),
        3: trueskill.Rating(),
        4: trueskill.Rating(),
        5: trueskill.Rating(),
    }

    graph = np.zeros((N_PLAYERS, N_PLAYERS))  # graph represents the transition matrix
    graph[0, [1, 2]] = 1
    graph[1, 3] = 1
    graph[2, 3] = 1
    graph[3, [4, 5]] = 1

    data = []
    for i in range(N_PLAYERS):
        ch = children(graph, i)
        for j in ch:
            # Sample the number of games between this pair
            K = np.random.choice(range(5), p=[0.3, 0.1, 0.1, 0.1, 0.4])
            for k in range(K):
                data.append([i, j])

    data = np.array(data)
    #print "Games:\n{}".format(data)
    #print "Size {}x{}".format(data.shape[0], data.shape[1])

    n_games = []

    for i in range(N_PLAYERS):
        s = 0
        for j in data:
            if j[0] == i or j[1] == i:
                s += 1
        n_games.append(s)

    print "# of games p/ gamer: {}".format(n_games)

    for w, l in data:
        PLAYERS[w], PLAYERS[l] = trueskill.rate_1vs1(PLAYERS[w], PLAYERS[l])

    domain = np.linspace(0, 40, 200)

    for p in PLAYERS.keys():
        y = [norm.pdf(x, PLAYERS[p].mu, PLAYERS[p].sigma) for x in domain]
        plt.plot(domain, y, label="Player {}".format(p))
        print "Player #{}: mu={}, sigma={}".format(p, PLAYERS[p].mu, PLAYERS[p].sigma)
    plt.legend()
    plt.show()

trueskill_demo()
