#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sistema de captura amostral de tweets de candidatos
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
import numpy as np
import datetime as dt
import os
import time
import json
import traceback
from pathlib import Path

import xavy.twitter as tw


def extract_twitter_username(series, lower=True):
    """
    Given a Series that contains the URL 
    or the username of twitter accounts,
    extract the username.
    
    If `lower` is True, return lower case
    username (Twitter usernames are not 
    case-sensitive).
    
    Returns a series with same index as 
    the input. NaN is returned when no
    unsername is found.
    """
    
    username =  series.str.extract('(?:[Tt]wit+er\.com(?:\.br)?/@?|@)(\w+)')[0]
    if lower is True:
        username = username.str.lower()
    
    return username


def request_twitter_user_info(df, username_col='twitter_username', cand_id_col='SQ_CANDIDATO', requests_per_window=900):
    """
    Collect twitter user info using Twitter
    API v.1.1.
    
    Parameters
    ----------
    df : DataFrame
        Table containing the twitter username 
        and a candidate identifier (e.g. 
        'SQ_CANDIDATO').
    username_col : str
        Name of the column containing the 
        twitter usernames.
    cand_id_col : str
        Column identifying the candidate.
    requests_per_window : int
        Maximum number of requests allowed
        by the API in a 15-minute window.
    
    Returns
    -------
    response_df : DataFrame
        DataFrame with all the information
        provided by the API, for each 
        username found, along with the 
        associated candidate ID.
    """
    
    # Look for twitter IDs with API:
    w_username = df[username_col].dropna().drop_duplicates()
    response = tw.lookup(w_username, requests_per_window=requests_per_window)
    
    # Build DataFrame with responses:
    response_df = pd.DataFrame(response['data'])
    # Join SQ_CANDIDATO to twitter data:
    response_df['lower_name'] = response_df['username'].str.lower()
    cand_ids = df.set_index(username_col)[cand_id_col].astype(str)
    response_df = response_df.join(cand_ids, on='lower_name')
    
    # Expand dict Series (nested data):
    for col in response_df.columns:
        if type(response_df[col].iloc[0]) is dict:
            nested_df = pd.DataFrame(list(response_df[col]), index=response_df.index)
            response_df = response_df.join(nested_df)
            response_df.drop(col, axis=1, inplace=True)
    
    return response_df


def append_mentions_page(mentions, url, parameters):
    """
    Update `mentions` and `parameters` in place
    by appending twitter API /2/users/:id/mentions
    responses and getting next page token.
    
    Parameters
    ----------
    mentions : dict
        Concatenated data from multiple API
        calls.
    url : str
        Twitter API URL 
        ('https://api.twitter.com/2/users/:id/mentions')
    parameters : dict
        API call parameters, which may include a 
        'pagination_token' key. The latter is 
        updated in place if there is a 'next_page' 
        in the API response.
    
    Returns
    -------
    appended_mentions : dict
        Dict containing the data from `mentions`, 
        appended by the data from the API call.
    is_new_page : bool
        True if there is another page after
        the current one, and False otherwise.
    """
    
    extra = tw.request_twitter_api(url, parameters)
    mentions = tw.concat_json_pages(mentions, extra)
    if 'next_token' in extra['meta']:
        parameters.update({'pagination_token': extra['meta']['next_token']})
        return mentions, True
    
    return mentions, False


def mentions_to_df(mentions, user_id):
    """
    Parse JSON structure containing twitter 
    mentions to a user into a DataFrame.

    Parameters
    ----------
    mentions : dict
        Twitter API response to 
        /2/users/:id/mentions endpoint.
    user_id : str ir int
        Twitter ID of the mentioned user.
    
    Returns
    -------
    mentions_df : DataFrame
        Data from the API parsed into a 
        DataFrame, with some extra columns.
    """
    
    # Cria DataFrame de tweets mencionando usuário:
    mentions_df = pd.DataFrame(mentions['data'])
    n_mentions  = len(mentions_df)
    
    # Adiciona coluna 'in reply...' se não existir (acho que isso acontece quando nenhuma das menções é reply):
    if 'in_reply_to_user_id' not in mentions_df.columns:
        mentions_df['in_reply_to_user_id'] = np.NaN
    
    # Junta informações sobre o autor da menção:
    participants_df = pd.DataFrame(mentions['includes']['users'])[['id', 'name', 'username']].drop_duplicates()
    author_fields = {'name':'author_name', 'username':'author_username'}
    mentions_df = mentions_df.join(participants_df.rename(author_fields, axis=1).set_index('id'), on='author_id')
    assert(len(mentions_df) == n_mentions), 'Author info join increased number of mentions. This is wrong.'
    
    # Adiciona link p/ o tweet:
    mentions_df['tweet_url'] = 'https://www.twitter.com/' + mentions_df['author_username'] + '/status/' + mentions_df['id'].astype(str)
    
    # Parseia data (se existir):
    if 'created_at' in mentions_df.columns:
        mentions_df['created_at'] = pd.to_datetime(mentions_df['created_at'])
    
    # Conta número de usuários mencionados:
    if 'entities' in mentions_df.columns:
        mentions_df['n_mentions'] = mentions_df['entities'].apply(lambda s: len(s['mentions']))
    
    # Identifica reply direto:
    mentions_df['direct_reply'] = (mentions_df['in_reply_to_user_id'] == str(user_id)).astype(int)
    
    # Info da captura:
    
    
    return mentions_df


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


def get_mentions_in_period(user_id, start_time, end_time, max_pages=20, max_results=100, requests_per_window=450, verbose=True, bsb_time=True):
    """
    Get mentions to specified user within a period of time.
    
    Parameters
    ----------
    user_id : int
        Twitter user ID to check mentions for.
    start_time : str or datetime
        Beginning of the period in which to look for mentions. 
        If str, must be in the format ''%Y-%m-%dT%H:%M:%S'.
    end_time : str or datetime
        End of the period in which to look for mentions. 
        If str, must be in the format ''%Y-%m-%dT%H:%M:%S'.
    max_pages : int
        Maximum number of pages to go through when a paginated
        result is returned. Note that the API only checks and 
        returns the 800 most recent tweets.
    max_results : int
        Maximum number of results to return in each API call, 
        that is, in each page.
    requests_per_window : int
        Maximum number of calls in a 15-minute window allowed 
        by the API. Each call for a page is delayed by the 
        appropriate amount of time to avoid reaching this 
        limit.
    verbose : bool
        Whether to print page numbers as going through the
        pagination.
    bsb_time : bool
        Whether `start_time` and `end_time` are given at 
        Brasilia local time (UTC-3).
    
    Returns
    -------
    mentions : dict
        The API response, containing the tweets mentioning the 
        user `user_id`, after concatenating the pages.
    """
    
    # Hard-coded:
    url_template = 'https://api.twitter.com/2/users/{}/mentions'
    params = {'tweet.fields': ['created_at'], 'expansions':['author_id', 'in_reply_to_user_id', 'entities.mentions.username']}
    
    # Prepate input:
    sleep_time = tw.compute_sleep_time(requests_per_window)
    url = url_template.format(user_id)
    start_utc = parse_utc_time(start_time, bsb2utc=bsb_time).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_utc   = parse_utc_time(end_time, bsb2utc=bsb_time).strftime('%Y-%m-%dT%H:%M:%SZ')
    params.update({'max_results': max_results, 'start_time': start_utc, 'end_time': end_utc})
    
    mentions = {}
    # First capture:
    time.sleep(sleep_time)
    mentions, get_next_page = append_mentions_page(mentions, url, params)
    page_num = 1
    if verbose is True:
        print(page_num, end=' ')
    # Go through pagination:
    while get_next_page is True and page_num < max_pages:
        time.sleep(sleep_time)
        mentions, get_next_page = append_mentions_page(mentions, url, params)
        page_num += 1
        if verbose is True:
            print(page_num, end=' ')
    
    return mentions


def compute_time_period(start_time, end_time, time_fmt='%Y-%m-%dT%H:%M:%S'):
    """
    Return the time interval in hours (float) between the
    `start_time` and `end_time` (both str or datetime).
    If str, the input should be provided in the `time_fmt`
    format.
    """
    return (parse_utc_time(end_time, time_fmt) - parse_utc_time(start_time, time_fmt)).total_seconds() / 3600


def capture_stats(mentions, start_time, end_time, time_fmt='%Y-%m-%dT%H:%M:%S', max_mentions=800):
    """
    Compute statistical information about the
    response of an API mentions request.
    
    Parameters
    ----------
    mentions : dict
        Response from an API call, as returned by 
        the `get_mentions_in_period` function.
    start_time : str or datetime
        Start of the time period requested for the
        capture with `get_mentions_in_period`.
    end_time : str or datetime
        End of the time period requested for the
        capture with `get_mentions_in_period`.
    time_fmt : str
        Format of `start_time` and `end_time`.
    max_mentions : int
        Maximum number of recent tweets returned by
        the API. NOTE THAT THE CAPTURE END TIME 
        SHOULD BE THE CURRENT TIME FOR THE STATISTICS
        TO BE RIGHT.
        
    Returns
    -------
    n_mentions : int
    n_errors : int
    time_window : float
    collected_time : float
    t_win_weight : float
    """
    
    # Request stats:
    n_mentions  = np.sum(mentions['meta']['result_count'])
    if 'errors' not in mentions.keys():
        n_errors = 0
    else:
        n_errors    = len(mentions['errors'])

    # The time period expected to be covered by the request:
    time_window = compute_time_period(start_time, end_time, time_fmt)
    # Actual time period covered:
    collected_time = compute_time_period(mentions['data'][-1]['created_at'], mentions['data'][0]['created_at'], time_fmt='%Y-%m-%dT%H:%M:%S.000Z')
    # Compute statistical weight of the tweet to represent the expected time period:
    if n_mentions >= max_mentions:
        t_win_weight = time_window / collected_time
    else:
        t_win_weight = 1.0
    
    return n_mentions, n_errors, time_window, collected_time, t_win_weight


def get_last_mentions(user_id, last_hours=6, verbose=True):
    """
    Capture mentions to a twitter user in the last couple of hours.
    
    Parameters
    ----------
    user_id : int
        Twitter user ID to look mentions for.
    last_hours : float
        Number of hours in the past, from current time, to look 
        for mentions. Remember that the API returns at most 
        800 most recent tweets.
    
    Returns
    -------
    mentions_df : DataFrame
        Table containing the tweets mentioning `user_id` in the 
        `last_hours`, along with the author info and capture 
        process stats.
    """
    
    # Set time landmarks:
    end_time   = dt.datetime.now()
    start_time = end_time - dt.timedelta(hours=last_hours)
    
    # Capture the mentions:
    mentions = get_mentions_in_period(user_id, start_time, end_time, verbose=verbose)
    
    # Exit if there is no data:
    if mentions['meta']['result_count'] == 0:
        return None, {'batch_start': start_time, 'batch_end': end_time, 'batch_tweets': 0, 'batch_errors': 0}
    
    # Build the DataFrame:
    m_df = mentions_to_df(mentions, user_id)
    m_df['batch_user']  = user_id
    m_df['batch_start'] = start_time
    m_df['batch_end']   = end_time
    m_df['batch_tweets'], m_df['batch_errors'], m_df['target_t_win'], m_df['actual_t_win'], m_df['t_win_weight'] = capture_stats(mentions, start_time, end_time)
    
    return m_df, {'batch_start': start_time, 'batch_end': end_time, 'batch_tweets': m_df.iloc[0]['batch_tweets'], 'batch_errors': m_df.iloc[0]['batch_errors']}


def todays_tweet_limit(curr_level, cap_renew_date, tweet_cap=2000000, safety_buffer=100):
    """
    Compute maximum number of tweets that should be captured
    per day given the current capture quota usage.
    
    Parameters
    ----------
    curr_level : int
        Number of tweets already captured.
    cap_renew_date : str
        Date when the usage cap resets, in format '%Y-%m-%d'.
        Check https://developer.twitter.com/en/portal/dashboard.
    tweet_cap : int
        Monthly tweet cap.
        Check https://developer.twitter.com/en/portal/dashboard.
    safety_buffer : int
        Decrement in the number of tweets to be captured per pay,
        to avoid errors.
    
    Returns
    -------
    todays_lim : int
        Maximum number of tweets that should be captured today,
        assuming a constant rate up to cap renew date.
    """
    today = dt.date.today()
    renew = dt.date(*(int(x) for x in cap_renew_date.split('-')))
    days_to_renew = (renew - today).days
    if days_to_renew <=0:
        raise Exception('{} reached cap renew date {}: reset `cap_renew_date` to new date.'.format(today, renew))
    todays_lim    = int((tweet_cap - curr_level) / days_to_renew - safety_buffer)
    
    return todays_lim


def todays_n_cands(curr_level, cap_renew_date, avg_tweets, tweet_cap=2000000, tweets_buffer=100):
    """
    Compute the number of candidates to look mentions for in the
    period of one day.
    
    Parameters
    ----------
    curr_level : int
        Number of tweets already captured.
    cap_renew_date : str
        Date when the usage cap resets, in format '%Y-%m-%d'.
        Check https://developer.twitter.com/en/portal/dashboard.
    avg_tweets : float
        Average number of tweets mentioning a candidate in the 
        capture period.
    tweet_cap : int
        Monthly tweet cap.
        Check https://developer.twitter.com/en/portal/dashboard.
    tweets_buffer : int
        Decrement in the number of tweets to be captured per pay,
        to avoid errors.
    
    Returns
    -------
    n_cands : int
        Number of candidates that can have their mentions captured.
    """
    
    return int(todays_tweet_limit(curr_level, cap_renew_date, tweet_cap, tweets_buffer) / avg_tweets)


def batch_n_cands(curr_level, cap_renew_date, avg_tweets, capture_period, max_cands, tweet_cap=2000000, tweets_buffer=100):
    """
    Compute the number of candidates to look mentions for in 
    a capture batch.
    
    Parameters
    ----------
    curr_level : int
        Number of tweets already captured.
    cap_renew_date : str
        Date when the usage cap resets, in format '%Y-%m-%d'.
        Check https://developer.twitter.com/en/portal/dashboard.
    avg_tweets : float
        Average number of tweets mentioning a candidate in the 
        capture period.
    capture_period : int
        Capture window size for each user, in hours. 
    max_cands : int
        Clip the number of candidates to this value, e.g. to avoid
        sampling more than the total number of candidates.
    tweet_cap : int
        Monthly tweet cap.
        Check https://developer.twitter.com/en/portal/dashboard.
    tweets_buffer : int
        Decrement in the number of tweets to be captured per pay,
        to avoid errors.
    
    Returns
    -------
    n_cands : int
        Number of candidates that can have their mentions captured.
    """
    
    day_n_cands   = todays_n_cands(curr_level, cap_renew_date, avg_tweets, tweet_cap, tweets_buffer)
    n_cands = int(day_n_cands / (24 / capture_period))
    
    return n_cands


def read_config(filename='../tweets/tweet_capture_config.json'):
    """
    Read JSON from `filename` (str).
    """
    with open(filename, 'r') as f:
        config = json.load(f)
    return config


def write_config(config, filename='../tweets/tweet_capture_config.json'):
    """
    Write `config` (dict) to `filename` (str) as JSON.
    """
    with open(filename, 'w') as f:
        json.dump(config, f, indent=1)


def program_batch_capture(twitter_df, n_cands, previous_df=None, start_time=None, time_fmt='%Y-%m-%dT%H:%M:%S', random_state=None):
    """
    Generate DataFrame with a schedule for capturing data from 
    randomly sampled candidates, organized in batches spaced 
    during the day.
    
    Parameters
    ----------
    twitter_df : DataFrame
        Table with column 'id' containing all the candidates'
        Twitter IDs.
    n_cands : int
        Number of candidates to randomly select for today's 
        capture.
    previous_df : DataFrame
        Capture list from the previous batch. These user IDs 
        sre removed from the set before sampling, to avoid 
        data overlap.
    start_time : str, datetime or None
        Datetime to schedule the capture to. If str, in format 
        given by `time_fmt`. If None, get the current date.
    time_fmt : str
        Format of `start_time`, if provided as str.
    random_state : int or None
        Seed for randomly selecting candidates. Use None for 
        random seed.
    
    Returns
    -------
    df : DataFrame
        Table with candidates to capture, their batches and 
        the time they should be captured.
    """
    
    # Get current date if needed:
    if start_time is None:
        start_time = dt.datetime.now()
    else: 
        start_time = parse_utc_time(start_time, time_fmt, bsb2utc=False)
        
    # Select not repeating sample:
    if previous_df is None:
        no_repeat = twitter_df
    else:
        no_repeat = twitter_df.loc[~twitter_df['id'].isin(previous_df['id'])]
    
    # Randomly select candidates:
    daily_capture_df = no_repeat['id'].sample(n_cands, weights=no_repeat['sample_weight'], random_state=random_state).reset_index()
    daily_capture_df.rename({'index':'cand_id_pos'}, axis=1, inplace=True)
    
    # Prepare batch information:
    daily_capture_df['batch_size'] = n_cands
    daily_capture_df['batch_time'] = start_time
    daily_capture_df['status'] = 'queue'
    daily_capture_df['batch_start'] = np.NaN
    daily_capture_df['batch_end'] = np.NaN
    daily_capture_df['batch_tweets'] = np.NaN
    daily_capture_df['batch_errors'] = np.NaN
    
    return daily_capture_df


def print_to_file(error_log, filename):
    """
    Print `error_log` (str) into file with `filename` (str).
    """
    
    with open(filename, 'w') as f:
        f.write(error_log)


def gen_mentions_path(data_dir, batch_time, user_id):
    """
    Create filename and path for storing the data obtained 
    from a mentions capture.
    
    Parameters
    ----------
    data_dir : str
        Path to the root data dir (e.g. 'data/').
    batch_time : str or datetime
        Batch time for identification purposes. If str, in
        format '%Y-%m-%dT%H:%M:%S'.
    user_id : int
        ID of the user mentioned.
    
    Returns
    -------
    file_path : str
        Filename, including path, where to save captured 
        mentions.
    """
    return '{0:}{1:}/mentions_{1:}_{2:}.csv'.format(data_dir, parse_utc_time(batch_time, bsb2utc=False).strftime('%Y-%m-%dT%H:%M:%S'), user_id)


def make_necessary_dirs(filename):
    """
    Create directories in the path to `filename` (str), if necessary.
    """
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def run_batch_capture(batch_time, twitter_df, config, previous_batch=None, verbose=True, no_protected=False):
    """
    Randomly select candidates and capture twitter mentions
    to them.
    
    Parameters
    ----------
    batch_time : str or datetime
        Approximately the current time, to serve only as 
        an identifier of the batch.
    twitter_df : DataFrame
        Table of all twitter IDs, to sample from.
    config : dict
        Capture process configuration, including capture
        time window, folders for saving data and logs, 
        request rates.
    previous_batch : DataFrame or None
        If provided, do not sample IDs present in 
        `previous_batch`, to avoid data overlap.
    verbose : bool
        Whether to print capture counts ans status.
    no_protected : bool
        Whether to remove protected accounts from sampling.
        I think protected accounts do not return replies, 
        but they still return mentions.
    
    Returns
    -------
    batch_df : DataFrame
        List of sampled IDs, along with information about
        their capture.
    """
    
    # Parse datetime to str:
    if type(batch_time) is dt.datetime:
        batch_time = batch_time.strftime('%Y-%m-%dT%H:%M:%S')
    
    # Filter out protected twitter accounts is requested:
    if no_protected is True:
        ids_df = twitter_df.loc[twitter_df['protected'] == False]
    else:
        ids_df = twitter_df
        
    # Create batch of IDs to capture:
    tot_cands = len(twitter_df)
    n_cands   = batch_n_cands(config['curr_level'], config['cap_renew_date'], config['avg_tweets_per_cand'], config['capture_period'], tot_cands)
    batch_df  = program_batch_capture(ids_df, n_cands, previous_batch, start_time=batch_time)

    # Log batch data:
    batch_df.to_csv('{}capture_{}.csv'.format(config['log_dir'], batch_time), index=False)

    if verbose is True:
        print('  ', end='', flush=True)
    
    # Loop over IDs to capture:
    for i in batch_df.index.values:

        try:
            # Capture data:
            mentions_df, stats = get_last_mentions(batch_df.loc[i, 'id'], config['capture_period'], verbose=False)
            # Log capture statistics:
            for name, stat in stats.items():
                batch_df.loc[i, name] = stat
            # Save captured mentions:
            if mentions_df is not None:
                # Add column identifying the batch:
                mentions_df['batch_time'] = batch_time
                # Save:
                filename = gen_mentions_path(config['data_dir'], batch_time, batch_df.loc[i, 'id'])
                make_necessary_dirs(filename)
                mentions_df.to_csv(filename, index=False)      
            status = 'ok'
        except:
            # Record error:
            tb = traceback.format_exc()
            print_to_file(tb, '{}{}_i-{:05d}_id-{}.log'.format(config['error_dir'], batch_time, i, batch_df.loc[i, 'id']))
            status = 'error'
        
        finally: 
            # Log batch data:
            batch_df.loc[i, 'status'] = status
            batch_df.to_csv('{}capture_{}.csv'.format(config['log_dir'], batch_time), index=False)
            if verbose is True:
                print('{}: {}'.format(i, status), end=', ', flush=True)
    
    print('')
    return batch_df


def sum_batch_tweets(batch_df):
    tot_tweets = int(batch_df.loc[batch_df['status'] == 'ok', 'batch_tweets'].sum())
    return tot_tweets


def compute_avg_tweets(batch_df):
    """
    Compute the average number of tweets from 
    the capture batch and save it to the 
    config file.
    """
    
    # Compute average:
    avg_tweets = batch_df.loc[batch_df['status'] == 'ok', 'batch_tweets'].mean()
    avg_tweets = np.round(avg_tweets, 3)
    
    return avg_tweets
    

def next_batch_time(capture_period, ini_date='2022-08-12T00:00:00', date_fmt='%Y-%m-%dT%H:%M:%S'):
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


def load_saved_mentions(data_dir):
    result_df = pd.concat([pd.read_csv(f, dtype={'in_reply_to_user_id':str}) for f in Path(data_dir).rglob('*.csv')], ignore_index=True)
    return result_df


def log_print(string, start=False):
    print('{} {}: {}'.format('*' if start else ' ', dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), string), flush=True)


def update_current_level(new_tweets, config):
    
    config['curr_level'] += new_tweets
    write_config(config)


def update_cap_renew_date(config, ini_date):

    # Parse cap renew date:
    utc_renew_time = parse_utc_time(config['cap_renew_date'] + 'T00:00:00', bsb2utc=False)
    # Get time of the next batch:
    next_time = next_batch_time(config['capture_period'], ini_date=ini_date)
    
    # Update cap renew date to next month if necessary:
    if next_time >= utc_renew_time:
        new_renew_time = utc_renew_time + dt.timedelta(days=30)
        new_renew_time = new_renew_time + dt.timedelta(days=utc_renew_time.day - new_renew_time.day)
    # Kepp current cap renew date:
    else:
        new_renew_time = utc_renew_time
    
    # Save new cap renew date:
    config['cap_renew_date'] = new_renew_time.strftime('%Y-%m-%d')
    write_config(config)


def driver():
    
    config = read_config()
    batch_time = config['batch_ref_time']
    batch_df = None
    while True:
        
        # Wait for next batch:
        batch_time = next_batch_time(config['capture_period'], ini_date=batch_time)
        log_print('Next batch at [{}]. Sleeping...'.format(batch_time))
        sleep_time = (batch_time - dt.datetime.now()).total_seconds()
        time.sleep(sleep_time)
        
        # Load data and config:
        log_print('Reload config and ID pool!', True)
        config = read_config()
        twitter_df = pd.read_csv(config['twitter_ids_file'])
        tot_cands = len(twitter_df)
        n_cands = batch_n_cands(config['curr_level'], config['cap_renew_date'], config['avg_tweets_per_cand'], config['capture_period'], tot_cands, config['tweet_cap'], config['tweets_buffer'])
        config_message = 'Batch config! # cands: {:d}, current level: {:d}, cap renew date: {}, avg. tweets p. cand: {:.3f}, capture period: {:.3f}'
        log_print(config_message.format(n_cands, config['curr_level'], config['cap_renew_date'], config['avg_tweets_per_cand'], config['capture_period']))
        
        # Run next batch:
        log_print('Running batch...')
        batch_df = run_batch_capture(batch_time, twitter_df, config, batch_df)
        tot_tweets = sum_batch_tweets(batch_df)
        avg_tweets = compute_avg_tweets(batch_df)
        log_print('Finished batch! Tweets captured: {:d}, Avg tweets: {:.3f}'.format(tot_tweets, avg_tweets))
        
        # Update config:
        update_current_level(tot_tweets, config)
        update_cap_renew_date(config, batch_time)
        log_print('Updated config! current level: {:d}, cap renew date: {}'.format(config['curr_level'], config['cap_renew_date']))


# If running this code as a script:
if __name__ == '__main__':

    driver()
