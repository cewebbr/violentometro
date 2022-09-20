#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agregador diário de contagens de tweets e de ataques
Copyright (C) 2022  Henrique S. Xavier
Contact: hsxavier@gmail.com

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import pandas as pd
from pathlib import Path
import numpy as np
import joblib
import json
import matplotlib.pyplot as pl
import datetime as dt
import time


def log_print(string, start=False):
    print('{} {}: {}'.format('*' if start else ' ', dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), string), flush=True)


def read_config(filename='../tweets/tweets2metric_config.json'):
    """
    Read JSON from `filename` (str).
    """
    with open(filename, 'r') as f:
        config = json.load(f)
    return config


def write_config(config, filename='../tweets/webpage_data.json'):
    """
    Write `config` (dict) to `filename` (str) as JSON.
    """
    with open(filename, 'w') as f:
        json.dump(config, f, indent=1)


def one2oneViolations(df, colIndex, colMultiples):
    """
    Returns the unique values in colMultiples for a fixed value in colIndex 
    (only for when the number of unique values is >1).
    """
    return df.groupby(colIndex)[colMultiples].unique().loc[df.groupby(colIndex)[colMultiples].nunique()>1]


def cross_join_dfs(df1, df2):
    """
    Cross join two DataFrame, i.e. combine all rows from
    the first to all rows from the second. The final index
    is reset.
    """
    
    # Set all indexes to the same values:
    df1.index = np.zeros_like(df1.index)
    df2.index = np.zeros_like(df2.index)
    # Cross-join and reset index:
    dfx = df1.join(df2).reset_index(drop=True)
    
    return dfx


def robust_load_csv(filename, low_memory=False, verbose=True, **kwargs):
    """
    Load a CSV file into a DataFrame using multiple strategies to 
    ensure the loading.
    
    Parameters
    ----------
    filename : str of Path
        Path to the CSV file.
    low_memory : bool
        Whether to guess column data types in order to use less memory
        when loading.
    verbose : bool
        Whether to print warning with the filename that failed the 
        basic loading strategy.
    
    Returns
    -------
    df : DataFrame
        Data from the CSV file.
    """
    
    try:
        # Leitura básica:
        df = pd.read_csv(filename, low_memory=low_memory, **kwargs)

    except pd.errors.ParserError:
        # Erro pode ser causado por carriage return. Nesse caso, tenta:
        if verbose is True:
            print('  !! Arquivo {} não foi aberto. Tentando evitar carriage return...'.format(filename))
        df = pd.read_csv(filename, low_memory=low_memory, lineterminator='\n', **kwargs) 

    return df


def load_concat_csv(data_path, file_pattern='*.csv', **kwargs):
    """
    Load all selected files into a single DataFrame.
    
    Parameters
    ----------
    data_path : str or Path
        Root folder where all CSV files are located,
        even if is subfolders.
    file_pattern : str
        Pattern of the files to load.
    
    Returns
    -------
    df : DataFrame
        All the data inside the CSV files, concatenated.
    """
    # List files:
    data_files = sorted(Path(data_path).rglob(file_pattern))

    # Load files:
    df = pd.concat([robust_load_csv(f, **kwargs) for f in data_files], ignore_index=True)
    
    return df


def select_direct_text_tweets(mentions_df):
    """
    Remove tweets containing only images and mentions (i.e. no text) 
    and select those clearly directed to the mentioned user (either 
    direct replies or mentioning only one user).
    
    Parameters
    ----------
    mentions_df : DataFrame
        Tweets, with columns 'text' (the tweet content), 'direcy_reply'
        (1 if  a reply to the mentioned user, 0 otherwise) and 
        'n_mentions' (number of users mentioned in the tweet).
    
    Returns
    -------
    direct_df : DataFrame
        A slice of `mentions_df`, containing only the tweets containing
        text clearly directed to the mentioned user.
    """
    
    # Ignore tweets containing only images:
    only_image_regex = '^(?:@[A-Za-z0-9_]{1,15} )+(?:https://t.co/[A-Za-z0-9]{10})?$'
    with_text_df = mentions_df.loc[~mentions_df['text'].str.contains(only_image_regex)]

    # Select only direct replies or unique mentions:
    direct_df = with_text_df.loc[(with_text_df['direct_reply'] == 1) | (with_text_df['n_mentions'] == 1)]
    
    return direct_df


def parse_df_datetimes(df, dt_cols, fmt=None):
    """
    Parse str to datetime in place.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with columns to parse.
    dt_cols : list of (str or int)
        Columns containing datetimes to parse.
    fmt : str
        Format of the date to parse.        
    """
    
    # Standardize input
    if type(dt_cols) in (str, int):
        dt_cols = [dt_cols]
    
    # Loop over columns to parse:
    for col in dt_cols:
        df[col] = pd.to_datetime(df[col], format=fmt)


def load_tweets_df(data_dir, utc2bsb=True):
    """
    Load all CSV files in `data_dir` containing tweets into
    a single DataFrame.
    
    Parameters
    ----------
    data_dir : str or Path
        Folder containing the tweets (even if in subfolders), 
        stored as rows in CSV files.
    utc2bsb : bool
        Whether to transform the 'created_at' field from 
        UTC to 'America/Sao_Paulo'.
    
    Returns
    -------
    tweets_df : DataFrame
        All tweets listed in the `data_dir` folder and 
        subfolders.
        - 'created_at', 'batch_start' and 'batch_end' 
          are parsed to datetime in Brasilia time zone.
    """
    # Load tweets:
    tweets_df = load_concat_csv('../tweets/scored/')

    # Parse dates:
    tweets_df['created_at'] = pd.to_datetime(tweets_df['created_at'], utc=True)
    if utc2bsb is True:
        tweets_df['created_at'] = tweets_df['created_at'].dt.tz_convert('America/Sao_Paulo')
    tweets_df['batch_start'] = pd.to_datetime(tweets_df['batch_start'])
    tweets_df['batch_end']   = pd.to_datetime(tweets_df['batch_end'])
    tweets_df['batch_time']  = pd.to_datetime(tweets_df['batch_time'])
    
    # Weight hate score by time window size, for aggregate statistics:
    tweets_df['w_hate_score'] = tweets_df['hate_score'] * tweets_df['t_win_weight']
    
    return tweets_df


def avg_stat_per_cand_day(tweets_df, stat_col, ids, batches_per_day=8):
    """
    Compute the average of a tweet statistic over candidates 
    (targets) and days.
    
    Parameters
    ----------
    tweets_df : DataFrame
        Tweets mentioning the candidates in a given period.
        Columns required: 'batch_user' and 'batch_time'.
    stat_col : str
        Column in `tweets_df` containing numbers to compute
        the average of.
    ids : set or list of ints
        IDs of the candidates to select from `tweets_df` 
        over which the average should be computed.
    batches_per_day : float
        Number of capture batches in a day.
        
    Returns
    -------
    avg : float
        The average of `stat_col` over the candidates and 
        the days.
    """
    
    return tweets_df.loc[tweets_df['batch_user'].isin(ids), stat_col].sum() / len(ids) / tweets_df['batch_time'].nunique() * batches_per_day


def stat_change_factor(tweets_df, stat_col, ids, ref_ids):
    """
    Compute how an average of a statistic changes when 
    the set of candidates are replaced.
    
    Parameters
    ----------
    tweets_df : DataFrame
        Tweets mentioning the candidates in a given period.
        Columns required: 'batch_user' and 'batch_time'.
    stat_col : str
        Column in `tweets_df` containing numbers to compute
        the average of.
    ids : set or list of ints
        IDs of the candidates to select from `tweets_df` 
        over which the average should be computed.
    ref_ids : set or list of ints
        IDs of the candidates to select from `tweets_df` 
        over which the reference average should be computed.
    
    Returns
    -------
    factor : float
        The division of the average for `ref_ids` over the 
        average for `ids`. How much the reference has a higher 
        average statistic over the sample.
    """
    
    return avg_stat_per_cand_day(tweets_df, stat_col, ref_ids) / avg_stat_per_cand_day(tweets_df, stat_col, ids)


def sel_ids(tweets_df, ids):
    """
    Return slice of `tweets_df` (DataFrame) targeting the users 
    listed in `ids` (ste or list of ints).
    """
    
    return tweets_df.loc[tweets_df['batch_user'].isin(ids)]


def batch_time_to_date(datetime_series, capture_period=3):
    """
    Convert a datetime Series representing time periods 
    to a date Series.
    
    Parameters
    ----------
    datetime_series : Series of datetimes
        Each entry is the end of a time period of length
        `capture_period`.
    capture_period : float
        Duration, in number of hours, of each time period 
        in `datetime_series`.
        
    Returns
    -------
    date_series : Series of dates
        The date of the day where each time period in 
        `datetime_series` is most present.
    """
    
    return (datetime_series - pd.DateOffset(hours=capture_period / 2)).dt.date


def sum_per_day(tweets_df, stat_col, batch_time_col='batch_time', capture_period=3):
    """
    Group tweets by batch date and sum the chosen column.
    
    Parameters
    ----------
    tweets_df : DataFrame
        Tweets, along with statistical values.
    stat_col : str
        Name of the column in `tweets_df` to sum over.
    batch_time_col : str
        Name of the column containing the end of the 
        period of the captured batch.
    capture_period : float
        Duration, in number of hours, of each time period 
        in `datetime_series`.    
    
    Returns
    -------
    daily_sum : Series
        The sum of `stat_col` over each batch date.
    """
    return tweets_df.groupby(batch_time_to_date(tweets_df[batch_time_col], capture_period))[stat_col].sum()


def hour_seasonal_factor(tweets_df):
    """
    Compute an average factor between the number of tweets
    at different hours of the day.
    
    Parameters
    ----------
    tweets_df : DataFrame
        Tweets captured, along with the columns 'batch_time'
        and 't_win_weight'.
    
    Returns
    -------
    hour_factor : Series of floats
        A factor that relates the average number of tweets 
        posted at a certain batch final hour with one another.
        The index are the hours (int).
    """
    
    hour_group = tweets_df.groupby(tweets_df['batch_time'].dt.hour)
    total_tweets_per_hour = hour_group['t_win_weight'].sum()
    return total_tweets_per_hour / total_tweets_per_hour.max()


def load_id_pool_sources(filename):
    """
    Load CSV file specifying during which capture period
    each list of the candidate's twitter IDs was used.
    
    Parameters
    ----------
    filename : str or Path
        Path to a CSV file containing the filenames 
        of lists of twitter IDs used along each 
        time period.
    
    Returns
    -------
    source_df : DataFrame
        Filenames used in each time period.
    """
    
    source_df = pd.read_csv(filename)
    source_df['last_batch'].fillna('2050-01-01T00:00:00', inplace=True)
    parse_df_datetimes(source_df, ['first_batch', 'last_batch'])
    
    return source_df


def load_ID_pools(root_dir, source_df):
    """
    Load lists of candidates' twitter IDs.
    
    Parameters
    ----------
    root_dir : str
        Folder (ending in '/') containing the lists 
        (CSV files) of candidates' twitter IDs.
    
    source_df : DataFrame
        Table with a column 'file' containing filenames 
        of the lists mentioned above.
        
    Returns
    -------
    id_dfs : dict of DataFrames
        The keys are the filenames and the values are 
        the DataFrames with the candidates' twitter 
        IDs and other characteristics.
    """
    id_dfs = {f: pd.read_csv(root_dir + f) for f in source_df['file']}
    return id_dfs


def load_batch_logs(logs_dir, source_df):
    """
    Load logs of tweet capture batches.
    
    Parameters
    ----------
    logs_dir : str or Path
        Path to the folder containing the logs of the capture 
        batches as CSV files. Each CSV file is a batch and each 
        row shows the batch time, the mentioned candidate's ID, 
        the amount of tweets captured, among other information.
    source_df : DataFrame
        Table specifying during which capture period each list 
        of the candidate's twitter IDs was used. The name of the 
        file used for sampling candidates in each batch is 
        added as a new column to the logs.
    
    Returns
    -------
    batch_df : DataFrame
        Information about the capture process of mentions to 
        each candidate and at each time.
    """
    
    # Load batch table:
    batch_df = load_concat_csv(logs_dir)
    parse_df_datetimes(batch_df, 'batch_time')

    # Join twitter IDs pool:
    batch_df = cross_join_dfs(batch_df, source_df)
    batch_df = batch_df.query('first_batch <= batch_time and batch_time <= last_batch')

    # Cada batch, identificado pelo horário, tem um único tamanho:
    assert len(one2oneViolations(batch_df, 'batch_time', 'batch_size')) == 0
    # O número de perfis analisados é igual ao 'batch_size':
    assert (batch_df['batch_time'].value_counts().sort_index() == batch_df.groupby('batch_time')['batch_size'].max().sort_index()).all()
    
    return batch_df


def fix_hate_target(tweets_df, target_model_file):
    """
    Use ML model to estimate the probability that the target 
    of the tweet mention is indeed the mentioned candidate 
    and not someone else, and weight the hate score by this 
    probability.
    
    Parameters
    ----------
    tweets_df : DataFrame
        Tweets mentioning candidates, already hate-scored 
        ('hate_score' column) and with column 't_win_weight' 
        specifying a weight for normalizing by the capture 
        time window.
    target_model_file : str or Path
        Path to a joblib file containing a scikit-learn 
        model to classify a tweet as being directed to 
        someone else other than the person mentioned.
    
    Returns
    -------
    proc_df : DataFrame
        Same as `tweets_df` but with extra columns:
        - 'final_prob', the probability that the candidate 
        was attacked by the tweet;
        - 'w_final_prob', the prob. above weighted by the 
        time window, so attacks can be counted as if the 
        time windows were all the same.
    """
    target_model = joblib.load(target_model_file)
    tweets_df['target_prob']  = target_model.predict_proba(tweets_df)[:, 0]
    tweets_df['final_prob']   = tweets_df['hate_score'] * tweets_df['target_prob']
    tweets_df['w_final_prob'] = tweets_df['final_prob'] * tweets_df['t_win_weight']
    return tweets_df


def etl_tweets_df(data_dir, target_model_file, bad_users):
    """
    Load tweets, infer the probability the candidate is the 
    tweet's target and remove unwanted users.
    
    Parameters
    ----------
    data_dir : str or Path
        Folder containing the tweets (even if in subfolders), 
        stored as rows in CSV files.
    target_model_file : str or Path
        Path to a joblib file containing a scikit-learn 
        model to classify a tweet as being directed to 
        someone else other than the person mentioned.
    bad_users : list of ints
        List of IDs to remove from the tweets (for not 
        belonging to the candidate, for instance).
    
    Returns
    -------
    tweets_df : DataFrame
        Table with one tweet per row, already rated for 
        hate and the candidate not being the target, 
        with tweets directed to `bad_users` removed.
    """
    
    # Load tweets:
    tweets_df = load_tweets_df(data_dir)

    # Identifica alvo do tweet:
    tweets_df = fix_hate_target(tweets_df, target_model_file)

    # Remove bad accounts:
    tweets_df = tweets_df.loc[~tweets_df['batch_user'].isin(bad_users)]
    
    return tweets_df


def build_different_pop_factors(tweets_df, id_ids, ref_ids, ref_start):
    """
    Compute factors to standardize samplings from different
    populations as if the sampling was the same.
    
    Parameters
    ----------
    tweets_df : DataFrame
        Tweets mentioning candidates, already target-hate-scored 
        ('w_final_prob' column) and with columns columns 
        'batch_time' (the datetime when the batch started) and 
        't_win_weight', specifying a weight for normalizing by the capture 
        time window.
    id_ids : dict of sets of ints
        Keys are the ID pool filenames, and values are sets of the IDs
        associated to each name, already filtered for a specific 
        group.
    ref_ids : set of ints
        One of the sets above (the one used as standard in terms of 
        average number of tweets and aggressions).
    ref_start : datetime
        When the ID pool that results in `ref_ids` started being used.
        
    Returns
    -------
    list_factors : DataFrame
        Index are ID pool filenames, and the columns are multiplying
        factors for the candidate-aggregated tweet and hate counts during 
        the periods where each ID pool filename was being used.
    """
    
    # Build factor to standardize statistics for different populations:
    renamer = {0:'n_tweets_list_factor', 1:'n_hate_list_factor'}
    last_list_df = tweets_df.loc[tweets_df['batch_time'] >= ref_start]
    list_factors = pd.DataFrame([{f: stat_change_factor(last_list_df, stat, id_ids[f], ref_ids) for f in id_ids.keys()} for stat in ['t_win_weight', 'w_final_prob']]).transpose().rename(renamer, axis=1)
    
    return list_factors


def aggregate_tweet_counts_over_candidates(selection_df, batch_df, ref_ids, list_factors, season_index, official_start='2022-08-16 03:30:00', capture_period=3):
    """
    Aggregate tweet and hate counts over candidates, creating 
    some time series.
    
    Parameters
    ----------
    selection_df : DataFrame
        Tweets DataFrame (one per row) already filtered to only 
        include direct replies or unique mentions to candidates 
        listeed in `ref_ids`.
    batch_df : DataFrame
        Log of the capture batches (one candidate and batch time 
        per row), with column 'status', 'batch_time' and 'id'.
    ref_ids : set of ints
        Twitter user IDs of those selected in `selection_df`. 
        Used to filter `batch_df`.
    list_factors : DataFrame
        Index are ID pool filenames, and the columns are multiplying
        factors for the candidate-aggregated tweet and hate counts during 
        the periods where each ID pool filename was being used.
    season_index : Series
        Index are the hours in the day (int) and values are relative 
        factors between them reflecting the average number of tweets 
        posted at each batch start time. Used to deseasonalize twitter
        counts so missing data can be properly imputed.
    official_start : str or datetime
        Datetime of the first batch that should have been captured 
        (i.e. the first batch schedule datetime after the start of 
        the campaign).
    capture_period : int
        Number of hours that each capture covers. This is also the time 
        interval between batches' start time.
    
    Returns
    -------
    capture_df : DataFrame
        Table whose index are batch trigger datetimes and the columns
        'final_n_tweets' and 'final_n_hate' show the total number 
        of tweets and agressions posted in that batch, estimated for 
        the whole population in `ref_ids`.
    """
    
    # Build DataFrame with all time intervals:
    good_batch_df = batch_df.loc[batch_df['id'].isin(ref_ids) & (batch_df['status'] == 'ok')]
    capture_df = pd.DataFrame(index=pd.date_range(official_start, good_batch_df['batch_time'].max(), freq=str(capture_period) + 'H'))
    n_captures = len(capture_df)

    # Count the number of candidates correctly monitored:
    n_cands = good_batch_df['batch_time'].value_counts() 
    n_cands.name = 'n_cands'
    capture_df = capture_df.join(n_cands).fillna(0)

    # Compute a factor for the estimate on the candidate population, from a sample:
    capture_df['pop_factor'] = len(ref_ids) / capture_df['n_cands']
    capture_df.loc[capture_df['pop_factor'] == np.inf, 'pop_factor'] = np.NaN

    # Compute the number of tweets for the population:
    capture_df['w_n_tweets'] = selection_df.groupby('batch_time')['t_win_weight'].sum()
    capture_df['w_n_hate']   = selection_df.groupby('batch_time')['w_final_prob'].sum()

    # Join factor for different populations: 
    capture_df = capture_df.join(good_batch_df.drop_duplicates(subset=['batch_time', 'file']).set_index('batch_time')['file'])
    capture_df = capture_df.join(list_factors, on='file')

    # Compute the final estimate for the population, when there is data for that batch:
    capture_df['pop_n_tweets'] = capture_df['pop_factor'] * capture_df['w_n_tweets'] * capture_df['n_tweets_list_factor']
    capture_df['pop_n_hate']   = capture_df['pop_factor'] * capture_df['w_n_hate'] * capture_df['n_hate_list_factor']

    # Impute missing values with time-seasonal interpolation:
    mov_window = int(24 / capture_period) - 1
    capture_df['hour'] = capture_df.index.hour
    capture_df = capture_df.join(season_index, on='hour')
    capture_df['imputer_n_tweets'] = (capture_df['pop_n_tweets'] / capture_df['season_index']).rolling(mov_window, center=True, closed='both').mean().interpolate().fillna(method='bfill') * capture_df['season_index']
    capture_df['imputer_n_hate']   = (capture_df['pop_n_hate'] / capture_df['season_index']).rolling(mov_window, center=True, closed='both').mean().interpolate().fillna(method='bfill') * capture_df['season_index']
    capture_df['final_n_tweets']   = capture_df['pop_n_tweets'].fillna(capture_df['imputer_n_tweets'])
    capture_df['final_n_hate']     = capture_df['pop_n_hate'].fillna(capture_df['imputer_n_hate'])

    assert len(capture_df) == n_captures
    
    return capture_df


def place_legend(loc, fontsize, fontcolor):
    """
    Place and format plot legend.
    """
    l = pl.legend(loc=loc, fontsize=fontsize)
    l.get_frame().set_alpha(None)
    l.get_frame().set_facecolor((0, 0, 0, 0.0))
    for text in l.get_texts():
        text.set_color(fontcolor)


def format_pt_ints(labels):
    """
    Fortmat labels returned by `ax.get_yticklabels()` 
    with periods splitting the thousands.
    """
    new_labels = ['{:,}'.format(int(t.get_text())).replace(',', '.') for t in labels]
    return new_labels


def plot_hate_daily_series(hate_series, tweet_series):
    """
    Returns a plot with 3 subplots, all daily time series:
    - The fraction of tweets that are violent;
    - The number of tweets that are violent;
    - The total number of tweets.
    """
    
    # Hard coded:
    ticksize     = 13
    labelsize    = 14 
    legendsize   = 14
    pad_factor   = 1.1
    frame_color  = '0.1'
    grid_color   = '0.85'
    legend_loc   = 'upper right'
    grid_axis    = 'x'
    monday_ticks = pd.date_range('2022-08-15', '2022-10-03', freq='7D')
    
    # Cria figura:
    f, axes = pl.subplots(3, 1, gridspec_kw={'height_ratios': [2, 2, 2]}, figsize=(10, 10))
    (a0, a1, a2) = axes

    # Plot da fração de tweets com agressões:
    pl.sca(a0)
    percent_series = (hate_series / tweet_series * 100)
    percent_series.plot(marker='o', color='#fdcb09', label='Fração violenta')
    # Format:
    pl.xticks(monday_ticks, [''] * len(monday_ticks))
    pl.ylim([percent_series.min() / pad_factor, percent_series.max() * pad_factor])
    pl.ylabel('%', fontsize=labelsize, color=frame_color)
    a0.spines['top'].set_visible(False)

    # Plot do total de tweets com agressões:
    pl.sca(a1)
    (hate_series).plot(marker='o', color='#5e6264', label='Tweets violentos')
    # Format:
    pl.xticks(monday_ticks, [''] * len(monday_ticks))
    pl.ylim([0, hate_series.max() * pad_factor])
    pl.ylabel('Número de tweets', fontsize=labelsize, color=frame_color)

    # Plot do total de tweets:
    pl.sca(a2)
    (tweet_series).plot(marker='o', color='#e5c493', label='Todos os tweets')
    # Format:
    pl.xticks(monday_ticks, monday_ticks.strftime('%d/%m'))
    pl.ylim([0, tweet_series.max() * pad_factor])
    pl.ylabel('Número de tweets', fontsize=labelsize, color=frame_color)

    # General format:
    for ax in axes:

        # Frames, ticks:
        ax.tick_params(color=frame_color, labelcolor=frame_color, labelsize=ticksize)
        ax.tick_params(axis='y', color=grid_color, length=5)
        for spine in ax.spines.values():
            spine.set_edgecolor(frame_color)
        for side in ['right', 'left']:
            ax.spines[side].set_visible(False)

        pl.sca(ax)
        pl.xlabel('')
        # Range:
        pl.xlim(['2022-08-15', '2022-10-03'])
        # Grid:
        pl.grid(axis=grid_axis, color=grid_color)
        # Legend:
        place_legend(legend_loc, legendsize, frame_color)
    
    # Formata números grandes no eixo:
    f.canvas.draw()
    for ax in (a1, a2):
        labels = format_pt_ints(ax.get_yticklabels())
        ax.set_yticks(ax.get_yticks()[:-1])
        ax.set_yticklabels(labels[:-1])
    
    pl.subplots_adjust(hspace=0)
    
    return f


def write_js(webpage_data, filename):
    content = "data = '" + json.dumps(webpage_data) + "'"
    with open(filename, 'w') as f:
        f.write(content)
        

def save_webpage_data(hate_series, fig_name, filename='../webpage/data/webpage_data.json'):
    """
    Write big numbers and a plot filename regarding the 
    violence on twitter to a JSON file.
    
    Parameters
    ----------
    hate_series : Series
        Time series with daily frequency containing the 
        number of attacks directed to candidates per day
        since the beginning of the campaign.
    fig_name : str
        Filename of the plot showing the time series.
    filename : str
        Path to the JSON file that will save the information
        above.
    """
    
    # Pega a data de ontem:
    yesterday = (pd.to_datetime('today') - pd.DateOffset(days=1)).date()
    
    # Cria JSON com números e nome da figura:
    webpage_data = {'total_counts': int(hate_series.sum() + 0.5),
                    'yesterday_counts': int(hate_series[yesterday] + 0.5),
                    'time_series_plot': fig_name,
                    'last_update': pd.to_datetime('today').strftime('%Y-%m-%dT%H:%M:%S')}
    
    # Salva JSON:
    write_js(webpage_data, filename)

    
def parse_utc_time(time_in, time_fmt='%Y-%m-%dT%H:%M:%S', bsb2utc=True):
    """
    Parse a (possibly) local time into UTC time.
    
    Parameters
    ----------
    time_in : str or datetime
        Time to parse to UTC datetime.
    time_fmt : str
        If `time_in` is str, this is used to parse it to datetime.
    bsb2utc : bool
        Whether to assume `time_in` is Brasilia local time and 
        convert it to UTC.
        
    Returns
    -------
    time_utc : datetime
        Time in UTC (assuming `time_in` is UTC and `bsb2utc` is
        False; or `time_in` is UTC-3 and `bsb2utc` is True).
    """
    # Parse str to datetime:
    if type(time_in) is str:
        time_dt = dt.datetime.strptime(time_in, time_fmt)
    else:
        time_dt = time_in
        
    # Convert Brazilia (UTC-3) to UTC:
    if bsb2utc is True:
        time_dt = time_dt + dt.timedelta(hours=3)
    
    return time_dt


def next_batch_time(capture_period, ini_date='2022-08-12T04:00:00', date_fmt='%Y-%m-%dT%H:%M:%S'):
    """
    Compute the datetime of the next batch from now.
    
    Parameters
    ----------
    capture_period : float
        Number of hours between each batch.
    ini_date : str or datetime
        Initial date (if str, in format `date_fmt`), from which
        the following batches are scheduled.
    date_fmt : str
        Format of `ini_date`.
    
    Returns
    -------
    next_date : datetime
        When to run the next batch capture
    """
    next_date = parse_utc_time(ini_date, bsb2utc=False)
    now_date  = dt.datetime.now()
    while next_date < now_date:
        next_date = next_date + dt.timedelta(hours=capture_period)
        
    return next_date


def analyse_tweets(config):
    """
    Load data listed under the appropriate `config` (dict) keys;
    process it to compute the total number of violent tweets yesterday
    and since the beginning of the elections; create a plot of time 
    series about these tweets; save everything to a PNG and a JSON file.
    """
    
    # Registro de pools de IDs utilizados:
    log_print('Loading list of twitter ID lists...', start=True)
    source_df = load_id_pool_sources(config['id_pool_sources'])   
    # Carrega todos os arquivos de pool de perfis de candidatos:
    log_print('Loading twitter ID lists...')
    id_dfs = load_ID_pools(config['id_pool_dir'], source_df)
    # Load capture logs:
    log_print('Loading capture logs...')
    batch_df = load_batch_logs(config['batch_logs_dir'], source_df)
    # Load tweets:
    log_print('Loading scored tweets...')
    tweets_df = etl_tweets_df(config['scored_tweets_dir'], config['target_model'], config['bad_users'])
    # Filter tweets directed to a single person:
    log_print('Select direct tweets only...')
    direct_df = select_direct_text_tweets(tweets_df)
    # Pesos para os diferentes pools de IDs ao calcular o agregado sobre os candidatos:
    log_print('Loading normalizing factors for different ID populations...')
    list_factors = pd.read_csv(config['pool_factor_file']).set_index('pool_filename')

    # Recorte aplicado aos dados:
    log_print('Preparing to select candidates...')
    query  = config['cand_sel_query']
    capture_period = config['capture_period']
    # IDs disponíveis em cada lista (removidos usuários errados):
    id_ids = {k:set(id_dfs[k].query(query)['id']) - set(config['bad_users']) for k in id_dfs.keys()}
    # IDS e data da última lista como referência:
    ref_ids   = id_ids[source_df['file'].iloc[-1]]
    
    # Seleciona tweets direcionados às candidaturas especificadas:
    log_print('Selecting candidates subgroup and removing bad users...')
    selection_df = sel_ids(direct_df, id_ids['twitter_ids_deputados_2022_v04.csv'])
    
    # Compute deseasonalyzing factor (to impute missing scheduled batches):
    log_print('Computing deseasonalizing factors...')
    season_index = hour_seasonal_factor(tweets_df)
    season_index.name = 'season_index'
    
    # Aggregate tweets over candidates to produce tweets counts and hate counts time series:
    log_print('Aggregating tweets to form batch time series...')
    capture_df = aggregate_tweet_counts_over_candidates(selection_df, batch_df, ref_ids, list_factors, season_index, config['official_start'], config['capture_period'])

    # Aggregate counts over time to create a series per day:
    log_print('Aggregating time series over the days...')
    yesterday    = (pd.to_datetime('today') - pd.DateOffset(days=1)).date()
    hate_series  = sum_per_day(capture_df.reset_index(), 'final_n_hate', 'index', capture_period).loc[:yesterday]
    tweet_series = sum_per_day(capture_df.reset_index(), 'final_n_tweets', 'index', capture_period).loc[:yesterday]

    # Cria e salva gráfico:
    log_print('Creating time series plot...')
    fig_name = config['time_series_plot'].format(pd.to_datetime('today').strftime('%Y-%m-%d'))
    fig = plot_hate_daily_series(hate_series, tweet_series)
    log_print('Saving time series plot...')
    fig.savefig(fig_name, transparent=True)
    
    # Write info to JSON:
    log_print('Saving resulting data to JSON...')
    save_webpage_data(hate_series, fig_name, config['webpage_json_file'])
    

def driver():
    
   
    # Read config:
    config = read_config('../tweets/tweets2metric_config.json')
    batch_time = config['analyse_ref_time']

    while True:
        # Wait for next batch:
        batch_time = next_batch_time(config['analyse_period'], ini_date=batch_time)
        log_print('Next analysis at [{}]. Sleeping...'.format(batch_time))
        sleep_time = (batch_time - dt.datetime.now()).total_seconds()
        time.sleep(sleep_time)

        # Read config:
        config = read_config('../tweets/tweets2metric_config.json')
        # Process data to produce summary stats and plots:
        analyse_tweets(config)
    

# If running this code as a script:
if __name__ == '__main__':
     
    driver()
