import trueskill
from matplotlib import pyplot as plt

# Head-to-head (1 vs. 1) match rule

# Most competition games follows 1:1 match rule. If your game does,
# just use _1vs1 shortcuts containing rate_1vs1() and quality_1vs1().
# These are very easy to use.

# First of all, we need 2 Rating objects:

r1 = trueskill.Rating()  # 1P's skill
r2 = trueskill.Rating()  # 2P's skill
e1 = trueskill.Rating()  # 1P's effort
e2 = trueskill.Rating()  # 2P's effort

# Then we can guess match quality which is equivalent with draw probability of
# this match using quality_1vs1():

#print('{:.1%} chance to draw'.format(trueskill.quality_1vs1(r1, r2)))

# After the game, TrueSkill recalculates their ratings by the game result.
# For example, if 1P beat 2P:

new_r1, new_e1, new_r2, new_e2 = trueskill.rate_extension_1vs1(r1, r2, e1, e2)
print("New Skill P1: {}".format(new_r1))
print("New Skill E1: {}".format(new_e1))
print("New Skill P2: {}".format(new_r2))
print("New Skill E2: {}".format(new_e2))

# Mu value follows player's win/draw/lose records. Higher value means higher
# game skill. And sigma value follows the number of games. Lower value means
# many game plays and higher rating confidence.

# So 1P, a winner's skill grew up from 25 to 29.396 but 2P, a loser's skill
# shrank to 20.604. And both sigma values became narrow about same magnitude.

# Of course, you can also handle a tie game with drawn=True:
#new_r1, new_r2 = trueskill.rate_1vs1(r1, r2, drawn=True)
#print(new_r1)
#print(new_r2)