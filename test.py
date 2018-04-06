from poissonOD import Rating, quality_1vs1, rate_1vs1

alice, bob = Rating(25), Rating(30)  # assign Alice and Bob's ratings

# if quality_1vs1(alice, bob) < 0.50:
#     print('This match seems to be not so fair')
alice, bob = rate_1vs1(alice, bob, 100)  # update the ratings after the match
print alice, bob