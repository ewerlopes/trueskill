import trueskill
from matplotlib import pyplot as plt

# Head-to-head (1 vs. 1) match rule

r1s = []
r2s = []

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

results = [0, 0, 0, 0, 0, 1, 1]
# for i in results:
#    if i:
#        r2, e2, r1, e1 = trueskill.rate_extension_1vs1(r2, r1, e1, e2, [0, 0])
#    else:
#        r1, e1, r2, e2 = trueskill.rate_extension_1vs1(r1, r2, e1, e2, [0, 0])
#
#    r1s.append(r1)
#    r2s.append(r2)
#    print("New Skill P1: {}\tE1: {}".format(r1, e1))
#    print("New Skill P2: {}\tE2: {}".format(r2, e2))
#    print ""



# Mu value follows player's win/draw/lose records. Higher value means higher
# game skill. And sigma value follows the number of games. Lower value means
# many game plays and higher rating confidence.

# So 1P, a winner's skill grew up from 25 to 29.396 but 2P, a loser's skill
# shrank to 20.604. And both sigma values became narrow about same magnitude.

# Of course, you can also handle a tie game with drawn=True:
for i in results:
    if i:
        r2, r1 = trueskill.rate_1vs1(r2, r1)#, drawn=True)
    else:
        r1, r2 = trueskill.rate_1vs1(r1, r2)

    r1s.append(r1.mu)
    r2s.append(r2.mu)

    print("New Skill P1: {}\tE1: {}".format(r1, e1))
    print("New Skill P2: {}\tE2: {}".format(r2, e2))
    print ""

plt.plot(r1s, label="r1")
plt.plot(r2s, label="r2")
plt.legend()
plt.show()

#print(new_r1)
#print(new_r2)