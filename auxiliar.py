import scipy.io as sio
import pandas as pd


def load_data(filename, verbose=False,
              header=('team1_ID', 'team2_ID', 'team_1_Score', 'team_2_Score')):

    if filename.split('.')[-1] == 'mat':
        # load data from mat
        match_outcome = sio.loadmat(filename)['matchOutcome']
        df = pd.DataFrame(data=match_outcome, columns=header)
    else:
        df = pd.read_csv(filename)
        df.columns = header

    if verbose:
        print 'Preview data:\n{}'.format(df.head())

    # NOTE: The Halo 2 data consists of a set of match outcomes comprising 6227 games for 1672 players.
    # We note there are negative scores for this data, so we add the absolute value of the minimal
    # score to all scores to use the data with all proposed models. (1st par of paper's section 5.1)

    min_score_t1 = df['team_1_Score'].min()
    min_score_t2 = df['team_2_Score'].min()
    abs_min = abs(min([min_score_t1, min_score_t2]))
    df['team_1_Score'] = df['team_1_Score'].apply(lambda x: x + abs_min)
    df['team_2_Score'] = df['team_2_Score'].apply(lambda x: x + abs_min)

    if verbose:
        print 'Preview data after excluding negatives:\n{}'.format(df.head())

    return df
