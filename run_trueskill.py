from trueskill import Rating, quality_1vs1, rate_1vs1
from collections import defaultdict
from matplotlib import pyplot as plt
from auxiliar import load_data


def run(df):
    team_skills = defaultdict(Rating)

    print df.head()

    print '\nTraining model...'

    # Update skills according to Poisson model
    for match in range(df.shape[0]):
        if not (match % 10):
            print 'Iter #{}'.format(match)

        team_ID_i = df['team1_ID'][match]
        team_ID_j = df['team2_ID'][match]

        team_skills[team_ID_i], team_skills[team_ID_j] = rate_1vs1(team_skills[team_ID_i], team_skills[team_ID_j])

    return team_skills

    alice, bob = Rating(25), Rating(30)  # assign Alice and Bob's ratings
    if quality_1vs1(alice, bob) < 0.50:
        print('This match seems to be not so fair')
    alice, bob = rate_1vs1(alice, bob)  # update the ratings after the match


if __name__ == '__main__':
    # using Halo2 dataset
    # data = load_data('dataset/Halo2Beta/matchOutcomeHalo.mat')

    # using simulated partial data
    data = load_data('dataset/partial_ordering/murphy_score.csv')
    skills = run(data)

    result = []
    for k in skills.keys():
        gauss = skills[k]
        sig = (1.0 / gauss.pi)  # variance
        mu = gauss.tau * sig  # mean
        result.append([k, mu, sig])

    print data
    plt.errorbar([r[0] for r in result], [r[1] for r in result], yerr=[r[2] for r in result], fmt='--o')
    plt.show()