from auxiliar import load_data
from collections import defaultdict
from poissonOD import Rating, rate_1vs1
from matplotlib import pyplot as plt
from scipy.special import iv
import math


def run(df):
    """
    Runs PoissonOD.
    :param df: the data as a pandas dataframe
    :return:
    """

    team_skills = defaultdict(list)

    print '\nTraining model...'
    # Update skills according to Poisson model
    for match in range(df.shape[0]):
        if not (match % 10):
            print 'Iter #{}'.format(match)

        team_ID_i = df['team1_ID'][match]
        team_ID_j = df['team2_ID'][match]
        s_i = df['team_1_Score'][match]
        s_j = df['team_2_Score'][match]

        if len(team_skills[team_ID_i]) == 0:
            team_skills[team_ID_i] = [Rating(), Rating()]

        if len(team_skills[team_ID_j]) == 0:
            team_skills[team_ID_j] = [Rating(), Rating()]

        o_i, d_j = rate_1vs1(team_skills[team_ID_i][0], team_skills[team_ID_j][1], s_i)
        o_j, d_i = rate_1vs1(team_skills[team_ID_j][0], team_skills[team_ID_i][1], s_j)

        team_skills[team_ID_i][0] = o_i
        team_skills[team_ID_i][1] = d_i
        team_skills[team_ID_j][0] = o_j
        team_skills[team_ID_j][1] = d_j

    print 'DONE!'
    return team_skills


def plot_results(sk, sk_type='offence', figsize=(4, 3)):
    sks_types = {'offence': 0, 'defence': 1}

    if sk_type not in sks_types.keys():
        print '{} is not a skill type. Skipping...'.format(sk_type)
        return

    result = []
    for k in skills.keys():
        gauss = sk[k][sks_types[sk_type]]
        print "Player #{} - {}".format(k, gauss)
        result.append([k, gauss.mu, gauss.sigma])
    fig = plt.figure(figsize=figsize)
    plt.errorbar([r[0] for r in result], [r[1] for r in result], yerr=[r[2] for r in result], fmt='--o')
    plt.title('{} skill'.format(sk_type))
    plt.ylabel('Skill value')
    plt.xlabel('Player ID')
    plt.tight_layout()
    plt.show()


def mae(pred, target):
    """Computes the mean absolute error"""
    m_err = 0
    N = len(pred)
    for p, t in zip(pred, target):
        m_err += (1.0/N) * abs(p-t)
    return m_err


def predict(lambda_i, lambda_j):
    """Computes the win probability"""

    p = 0
    for k in range(1, 101): # approximation to calculate the winProb.
        p += math.exp(-(lambda_i+lambda_j)) * (float(lambda_i)/float(lambda_j))**(k/2)\
             * iv(k, 2 *math.sqrt(lambda_i*lambda_j))
    return p


if __name__ == '__main__':
    # using Halo2 dataset
    # data = load_data('dataset/Halo2Beta/matchOutcomeHalo.mat')

    # using simulated partial data
    data = load_data('dataset/partial_ordering/murphy_score.csv')

    print 'Preview data:'
    print data.head()

    print 'Sum of points by team1_ID:'
    print data.groupby('team1_ID')['team_1_Score'].agg(['sum'])

    # calculate skills
    skills = run(data)

    plot_results(skills)
