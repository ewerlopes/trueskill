import scipy.io as sio
import pandas as pd
from collections import defaultdict
from poissonOD import Rating, rate_1vs1

# load data from mat
match_outcome = sio.loadmat('dataset/Halo2Beta/matchOutcomeHalo.mat')['matchOutcome']
df = pd.DataFrame(data=match_outcome, columns=['team1_ID', 'team2_ID', 'team_1_Score', 'team_2_Score'])

print 'Preview data:\n{}'.format(df.head())

team_skills = defaultdict(list)

# NOTE: The Halo 2 data consists of a set of match outcomes comprising 6227 games for 1672 players.
# We note there are negative scores for this data, so we add the absolute value of the minimal
# score to all scores to use the data with all proposed models. (1st par of paper's section 5.1)

min_score_t1 = df['team_1_Score'].min()
min_score_t2 = df['team_2_Score'].min()
abs_min = abs(min([min_score_t1, min_score_t2]))
df['team_1_Score'] = df['team_1_Score'].apply(lambda x: x + abs_min)
df['team_2_Score'] = df['team_2_Score'].apply(lambda x: x + abs_min)

print 'Preview data after excluding negatives:\n{}'.format(df.head())

print '\nTraining model...'
# Update skills according to Poisson model
for match in range(df.shape[0]):
    if not (match % 10):
        print 'Iter #{}'.format(match)

    teamID_i = df['team1_ID'][match]
    teamID_j = df['team2_ID'][match]
    s_i = df['team_1_Score'][match]
    s_j = df['team_2_Score'][match]

    if len(team_skills[teamID_i]) == 0:
        team_skills[teamID_i] = [Rating(), Rating()]

    if len(team_skills[teamID_j]) == 0:
        team_skills[teamID_j] = [Rating(), Rating()]

    o_i, d_j = rate_1vs1(team_skills[teamID_i][0], team_skills[teamID_j][1], s_i)
    o_j, d_i = rate_1vs1(team_skills[teamID_j][0], team_skills[teamID_j][1], s_j)

    team_skills[teamID_i][0] = o_i
    team_skills[teamID_i][1] = d_i
    team_skills[teamID_j][0] = o_j
    team_skills[teamID_j][1] = d_j