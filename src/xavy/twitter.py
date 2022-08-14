#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Funções facilitadoras do uso da API do twitter
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

import json
import requests
import datetime as dt
import time
import subprocess
import sys
from collections import defaultdict

from . import oauth


### SPECIFIC REQUESTS ###


def tweets_lookup(ids, include_entities=True, trim_user=False, list_missing=False, alt_text=True, include_card_uri=False, tweet_mode='extended', max_ids=100, requests_per_window=900):
    """
    Get the contents of the specified tweets.
    
    Input
    -----
    
    ids : str, int or list.
        IDs of the tweets to retrieve. It can be a list of 
        ints, strs or a single int or str.
    
    include_entities : bool
        Whether to include an entry in each tweet called
        'entities' that contains recognised entities such
        as 'hashtags', 'mentions', 'urls' and 'symbols'.
    
    trim_user : bool
        If True, return only the 'id' and 'id_str' of 
        the user. If False, return more information 
        (e.g. screen name, description, etc).
    
    list_missing : bool
        Whether to return a list of dicts where each
        dict is a found tweet (False), or a dict of 
        dicts with the keys are the tweet ids and the 
        values are the tweet contents. If the tweet 
        is missing, the associated value is null.
        
    alt_text : bool
        Whether to show alternative text (if provided) 
        associated to media in the tweet.
        
    include_card_uri : bool
        Something related to including AD cards info.
    
    max_ids : int
        Twitter API limit on the number of IDs.
        
    requests_per_window : int
        Maximum number of requests per 15 min 
        window allowed by the twitter API.
        
    Return
    ------
    
    return : list of dicts (or dict of dicts)
        The contents of the tweets.
    """
    
    
    # Compute sleep time:
    sleep_time = compute_sleep_time(requests_per_window)

    # Split users into batches:
    batches = batch_list(ids, max_ids)

    result = []
    for batch in batches:
        result += single_tweets_lookup(batch, include_entities, trim_user, list_missing, alt_text, include_card_uri, tweet_mode, max_ids)
        time.sleep(sleep_time)
    
    return result


def lookup(users, parameters={'user.fields':['description', 'created_at', 'protected', 'verified', 'public_metrics']},
           requests_per_window=900, max_users=100, version=2):
    """
    Return details about the specified `users`.
    
    Parameters
    ----------  
    users : str, int or list of these
        If `users` is a str or a list of str, assume it refers to 
        'screen_name's (usernames).
        If `users` is an int or a list of int, assume it refers to 
        'user_id's (IDs).
    parameters : dict
        Parameters of the API endpoint request, tipically the fields to
        be returned. Only used if `version` is 2.
    requests_per_window : int
        Maximum number of requests per 15 min window allowed by the 
        twitter API.
    max_users : int
        Maximum number allowed by the twitter API of users to be looked up.
    version : int
        Which API version to use: 1 or 2.
    
    Returns
    -------
    result : list of dicts.
        Each entry in the list is the info about one user in `users`.
    """  

    # Compute sleep time:
    sleep_time = compute_sleep_time(requests_per_window)

    # Standardize input:
    if type(users) in (str, int):
        users = [users]
    elif type(users) not in (list,):
        users = list(users)
    version = int(version)
    assert version in (1, 2), 'Unknown API version {}. Allowed values are 1 or 2.'.format(version)
    
    # Split users into batches:
    batches = batch_list(users, max_users)

    # Set API basic response format:
    result = [] if version == 1 else {}
    for batch in batches:
        # Get results for batch:
        partial = single_lookup(batch, parameters, max_users, version)
        # Join to results:
        result = concat_json_pages(result, partial)
        time.sleep(sleep_time)
    
    return result


def get_timeline(user, max_id=None, exclude_replies=True, include_rts=False, count=200, tweet_mode='extended', trim_user=True):
    """
    Get the latest tweets from a `user` timeline.
    
    Input
    -----
    
    user : str or int
        The screen name or the ID of the user.
        
    max_id : int or None
        The last tweet ID to return. If None, 
        return the tweets up to the latest one.
        
    exclude_replies : bool
        Whether or not to exclude tweets that
        are replies to other tweets.
        
    include_rts : bool
        Whether or not to return retweets as 
        well.
        
    count : int
        Maximum number of tweets to return.
    
    tweet_mode : str
        Whether or not to return the full 
        tweet (all characters) or not.
    
    trim_user : bool
        Whether to return only a selected 
        subset of user attributes.
        
    Return
    ------
    
    return : list of dicts
        The tweets.
    """
    
    # Set request parameters:
    parameters = {'max_id': max_id, 'exclude_replies': exclude_replies, 'include_rts': include_rts, 'tweet_mode': tweet_mode, 'trim_user': trim_user, 'count': count}
    parameters = append_user(parameters, user)
    if max_id == None:
        parameters.pop('max_id')
    
    endpoint = 'https://api.twitter.com/1.1/statuses/user_timeline.json'
    result = request_twitter_api(endpoint, parameters)
    
    return result


def get_timeline_since_date(user, min_date, exclude_replies=True, include_rts=False, tweet_mode='extended', trim_user=True, verbose=False):
    """
    Get all tweets from user since a minimum date.
    
    Input
    -----
    
    user : str or int
        The screen name or the ID of the user.
        
    min_date : str
        Oldest UTC date to cover, in format '%Y-%m-%d'.
        
    exclude_replies : bool
        Whether or not to exclude tweets that are 
        replies to other tweets.
        
    include_rts : bool
        Whether or not to return retweets as well.
        
    count : int
        Maximum number of tweets to return.
    
    tweet_mode : str
        Whether or not to return the full tweet (all 
        characters) or not.
    
    trim_user : bool
        Whether to return only a selected subset of 
        user attributes.
        
    verbose : bool
        Print messages along the way or not.
        
    Return
    ------
    
    return : list of dicts
        All tweets since at least date `min_date`
        (in UTC). It will usually include dates 
        beyond it.
    """
    
    # Compute sleep time to avoid violating API limits:
    sleep_time = compute_sleep_time(900)
    
    # Parse dates:
    min_date = dt.datetime.strptime(min_date, '%Y-%m-%d').date()

    # Get first tweets:
    result = get_timeline(user, max_id=None, exclude_replies=exclude_replies, include_rts=include_rts, tweet_mode=tweet_mode, trim_user=trim_user)
    # Get oldest tweet date:
    oldest_tweet = result[-1]
    oldest_date  = string_to_date(oldest_tweet['created_at']).date()
    if verbose:
        print('# tweets: {}   Oldest ID: {} ({})'.format(len(result), oldest_tweet['id'], oldest_date))
    
    while oldest_date >= min_date:
        
        # Wait to avoid violating API limits:
        if verbose:
            print('Sleep...')
        time.sleep(sleep_time)
        
        # Get more tweets:
        max_id = oldest_tweet['id'] - 1
        extra  = get_timeline(user, max_id, exclude_replies=exclude_replies, include_rts=include_rts, tweet_mode=tweet_mode, trim_user=trim_user)
        result = result + extra
        
        # Update oldest date:
        oldest_tweet = result[-1]
        oldest_date  = string_to_date(oldest_tweet['created_at']).date()

        if verbose:
            print('# tweets: {}   Oldest ID: {} ({})'.format(len(result), oldest_tweet['id'], oldest_date))
        
        # Security switch:
        if len(extra) == 0:
            if verbose:
                print('API is not returning any new results. Break!')
            break
    
    return result


def get_timeline_between_dates(user, min_date, max_date, exclude_replies=True, include_rts=False, tweet_mode='extended', trim_user=True, verbose=False):
    """
    Get all tweets from user between a minimum and 
    a maximum date (both inclusive).
    
    Input
    -----
    
    user : str or int
        The screen name or the ID of the user.
        
    min_date : str
        Oldest date to get (UTC), in format '%Y-%m-%d'.
    
    max_date : str
        Newest date to get (UTC), in format '%Y-%m-%d'.
    
    exclude_replies : bool
        Whether or not to exclude tweets that are 
        replies to other tweets.
        
    include_rts : bool
        Whether or not to return retweets as well.
        
    count : int
        Maximum number of tweets to return.
    
    tweet_mode : str
        Whether or not to return the full tweet (all 
        characters) or not.
    
    trim_user : bool
        Whether to return only a selected subset of 
        user attributes.
        
    verbose : bool
        Print messages along the way or not.
        
    Return
    ------
    
    return : list of dicts
        All tweets posted between the specified
        dates in UTC (both inclusive).
    """
    
    # Get all tweets covering the specified range:
    result = get_timeline_since_date(user, min_date, exclude_replies=exclude_replies, include_rts=include_rts, tweet_mode=tweet_mode, trim_user=trim_user, verbose=verbose)
    
    # Filter results to specified dates:
    if verbose:
        print('Filtering by dates...')
    result = list(filter(lambda tweet: compare_dates(tweet, min_date, max_date), result))

    if verbose:
        print('# tweets: {}'.format(len(result)), end='')
        if len(result) > 0:
            print('   {} -> {}'.format(string_to_date(result[-1]['created_at']), string_to_date(result[0]['created_at'])))
        else:
            print('')
            
    return result


def scrape_tweet_list(screen_name, min_date, max_date):
    """
    List all tweets from user `screen_name` (str)
    that were posted between dates `min_date` 
    (inclusive) and `max_date` (exclusive). 
    The dates are likely in UTC.

    Returns a list of str with links to the tweets.
    """
    
    # Set query:
    query = f'"from:{screen_name} since:{min_date} until:{max_date}"'
    
    # Get list of tweets:
    try:
        response = subprocess.check_output(['snscrape', 'twitter-search', query])
        tweets   = response.decode('utf-8').split('\n')[:-1] 
        return tweets
    
    except FileNotFoundError:
        raise Exception('This likely means you do not have snscrape installed. Please run `pip install snscrape`.')
    

### INTERMEDIARY REQUESTS ###


def single_tweets_lookup(ids, include_entities=True, trim_user=False, list_missing=False, alt_text=True, include_card_uri=False, tweet_mode='extended', max_ids=100):
    """
    Get the contents of the specified tweets.
    
    Input
    -----
    
    ids : str, int or list.
        IDs of the tweets to retrieve. It can be a list of 
        ints, strs or a single int or str.
    
    include_entities : bool
        Whether to include an entry in each tweet called
        'entities' that contains recognised entities such
        as 'hashtags', 'mentions', 'urls' and 'symbols'.
    
    trim_user : bool
        If True, return only the 'id' and 'id_str' of 
        the user. If False, return more information 
        (e.g. screen name, description, etc).
    
    list_missing : bool
        Whether to return a list of dicts where each
        dict is a found tweet (False), or a dict of 
        dicts with the keys are the tweet ids and the 
        values are the tweet contents. If the tweet 
        is missing, the associated value is null.
        
    alt_text : bool
        Whether to show alternative text (if provided) 
        associated to media in the tweet.
        
    include_card_uri : bool
        Something related to including AD cards info.
    
    max_ids : int
        Twitter API limit on the number of IDs.
        
    Return
    ------
    
    return : list of dicts (or dict of dicts)
        The contents of the tweets.
    """
    
    # Standardize input to a sequence:
    if type(ids) == str or type(ids) == int:
        ids = [ids]
    # Security check:
    assert len(ids) <= max_ids, 'The API limits the number of tweet IDs to {}.'.format(max_ids)
    # Transform sequence to CSV string:
    ids = list2csv(ids)
    
    # Make the request:
    parameters = {'id': ids, 'include_entities': include_entities, 'trim_user': trim_user, 'map': list_missing,
                  'include_ext_alt_text': alt_text, 'include_card_uri': include_card_uri, 'tweet_mode': tweet_mode}
    endpoint   = 'https://api.twitter.com/1.1/statuses/lookup.json'
    result = request_twitter_api(endpoint, parameters)
    
    return result


def single_lookup(users, parameters={'user.fields':['description', 'created_at', 'protected', 'verified', 'public_metrics']}, max_users=100, version=2):
    """
    Return details about the specified `users`.
    
    Input
    -----    
    users : str, int or list of these
        If `users` is a str or a list of str, assume it refers to usernames.
        If `users` is an int or a list of int, assume it refers to IDs.
    parameters : dict
        Parameters for the request, tipically the fields to be returned. 
        This is only used if `version` is 2.
    max_users : int
        This API service returns up to `max_users`. This value is used here
        only to alert the user if the number of users is larger. It does not
        afffect the output.
    version : int or str
        Twitter API version to use. Options are: 1 or 2.
    
    Returns
    -------
    result : list of dicts.
        Each entry in the list is the info about one user in `users`.
    """

    # Standardize input:
    if type(users) in (str, int):
        users = [users]
    elif type(users) not in (list,):
        users = list(users)
    
    if int(version) == 1:
        return single_lookup_v1(users, max_users)
    elif int(version) == 2:
        return single_lookup_v2(users, parameters, max_users)
    else:
        raise Exception('Uknknown API version {}. Options are 1 and 2.'.format(version))


def single_lookup_v2(users, parameters={'user.fields':['description', 'created_at', 'protected', 'verified', 'public_metrics']}, max_users=100):
    """
    Return details about the specified `users`, using twitter API version 2.
    
    Input
    -----    
    users : str, int or list of these
        If `users` is a str or a list of str, assume it refers to usernames.
        If `users` is an int or a list of int, assume it refers to IDs.
    parameters : dict
        Parameters for the request, tipically the fields to be returned.
    max_users : int
        This API service returns up to `max_users`. This value is used here
        only to alert the user if the number of users is larger. It does not
        afffect the output.
    
    Returns
    -------
    result : list of dicts.
        Each entry in the list is the info about one user in `users`.
    """
    
    # Standardize input:
    if type(users) in (str, int):
        users = [users]
    elif type(users) not in (list,):
        users = list(users)
    assert len(users) <= max_users, 'The API limits the number of users to {}.'.format(max_users)
    
    # Define endpoint por tipo de identificação:
    if (type(users[0]) is int) or (type(users[0]) is str and users[0].isdigit()):
        # Pega por ID:
        endpoint = 'https://api.twitter.com/2/users'
        userid   = 'ids'
    else:
        # Pega por nome de usuário:
        endpoint = 'https://api.twitter.com/2/users/by'
        userid   = 'usernames'
    
    # Add user to request parameters:
    if userid in parameters.keys():
        raise Exception("'{}' is already in `parameters` keys {}.".format(userid, list(parameters.keys())))
    fullpars = parameters.copy()
    fullpars.update({userid: users})
    
    # Make the request:
    result = request_twitter_api(endpoint, fullpars)
    
    return result


def single_lookup_v1(users, max_users=100):
    """
    Return details about the specified `users`.
    
    Input
    -----
    
    users : str, int or list of these
        If `users` is a str or a list of str, 
        assume it refers to 'screen_name's.
        If `users` is an int or a list of int,
        assume it refers to 'user_id's.
        
    Return
    ------
    
    result : list of dicts.
        Each entry in the list is the info
        about one user in `users`.
    """
    
    # Hard-coded:
    endpoint = 'https://api.twitter.com/1.1/users/lookup.json'

    # Standardize input:
    if type(users) in (str, int):
        users = [users]
    elif type(users) not in (list,):
        users = list(users)
    if type(users[0]) == int:
        name_input = False
    elif type(users[0]) == str:
        name_input = True
    else:
        raise Exception("Unknown `users` input type '{}'.".format(type(users[0])))

    assert len(users) <= max_users, 'The API limits the number of users to {}.'.format(max_users)
    
    # Create request parameters:
    users = list2csv(users)

    if name_input:
        parameters = {'screen_name': users}
    else:
        parameters = {'user_id': users}
        
    # Make the request:
    result = request_twitter_api(endpoint, parameters)
    
    return result


### BASICS ###


def concat_json_pages(j1, j2):
    """
    Concatenate contents of two JSONs `j1` 
    and `j2` by copying dict structures, 
    joining lists and other values as 
    elements of a list.
    """
    
    # If JSON is a list or a scalar:
    if type(j1) not in (dict, defaultdict) and type(j2) not in (dict, defaultdict):
        # Make both JSONs as lists:
        if type(j1) is not list:
            j1 = [j1]
        if type(j2) is not list:
            j2 = [j2]
        # Concatenate the lists:
        return j1 + j2
    
    # If they are dicts, apply recursion to each key:
    common_keys = list(set(j1.keys()) & set(j2.keys()))
    exclusive_1 = list(set(j1.keys()) - set(j2.keys()))
    exclusive_2 = list(set(j2.keys()) - set(j1.keys()))
    new_dict    = {k:concat_json_pages(j1[k], j2[k]) for k in common_keys}
    
    # Copy unique contents to output: 
    for k in exclusive_1:
        new_dict[k] = j1[k]
    for k in exclusive_2:
        new_dict[k] = j2[k]
    
    return new_dict


def batch_list(entries_list, batch_size):
    """
    Split the entries in `entries_list` (list or array)
    into a disjoint and complete list of arrays (or 
    lists), each containing at most `batch_size` entries.
    """
    
    n_entries = len(entries_list)
    
    entries_list = list(entries_list)
    
    batches = []
    for i in range(0, n_entries, batch_size):
        batches.append(entries_list[i: i + batch_size])
        
    return batches


def is_username(user):
    """
    Check if `user` is a str or int.
    This is used to know if `user` is
    a screen name or an ID.
    """
    if type(user) == int:
        return False
    elif type(user) == str:
        return True
    else:
        raise Exception("Unknown `user` input type '{}'.".format(type(user)))

    
def append_user(parameters, user):
    """
    Append (inplace) `user` (str or int) to 
    twitter API `parameters` (dict) under the 
    proper key, whether `user` is a screen name 
    or an ID.
    """
    
    if is_username(user):
        parameters['screen_name'] = user
    else:
        parameters['user_id'] = user

    return parameters


def compute_sleep_time(requests_per_window, max_frac=0.90, window_min=15):
    """
    Return a minimum time interval between requests to 
    avoid violating twitter API rules.
    
    Input
    -----
    
    requests_per_window : int
        Maximum number of requests allowed per time window.
        Check https://developer.twitter.com/en/docs/twitter-api/v1 
        for values.
    
    max_frac : float
        Factor from 0.0 to 1.0 to multiply `requests_per_window`
        to derive a safer maximum number of requests per window
        (with a buffer from the true limit).
    
    window_min : int or float
        Number of minutes correponding to one time window interval
        (tipically 15 minutes).
    
    Return
    ------
    
    sleep_time : float
        Number of seconds to wait between two consecutive
        API calls.
    """
    
    window_sec = window_min * 60
    max_req_per_window = int(max_frac * requests_per_window)
    sleep_time = window_sec / max_req_per_window
    
    return sleep_time


def compare_dates(tweet, min_date, max_date):
    """
    Return True if `tweet` (dict) happened is inside the 
    inclusive date interval defined by `min_date` and 
    `end_date` (str, given in format '%Y-%m-%d').  
    """

    # Parse dates:
    min_date = dt.datetime.strptime(min_date, '%Y-%m-%d').date()
    max_date = dt.datetime.strptime(max_date, '%Y-%m-%d').date()

    date = string_to_date(tweet['created_at']).date()       

    if (date <= max_date) and (date >= min_date):            
        return True
    else:
        return False


def string_to_date(date):
    """
    Convert a twitter created_at string to a datetime object.
    """

    date_time_str = date

    date_time_obj = dt.datetime.strptime(date_time_str, '%a %b %d %H:%M:%S +0000 %Y')

    return date_time_obj


def list2csv(x):
    """
    Transforms a list or tuple `x` into a 
    string with the elements in `x` 
    separated by commas.
    """
    return ','.join([str(entry) for entry in x])


def augment(url, parameters, credentials, http_method='GET'):
    """
    Create a URL for HTTP request with authentication 
    information. These are added as extra parameters.
    """
    
    # Create an object with the api key and secret key as attributes `key` and `secret`:
    consumer = oauth.OAuthConsumer(credentials['api_key'], credentials['api_secret_key'])
    # Create an object with the token and token secret as attributes; this object has to/from string methods: 
    token    = oauth.OAuthToken(credentials['access_token'], credentials['access_token_secret'])

    # Create an object with all the provided information plus OAuth version, timestamp and random number:
    oauth_request = oauth.OAuthRequest.from_consumer_and_token(consumer, token=token, http_method=http_method, http_url=url, parameters=parameters)
    # Create the attribute 'oauth_signature' in the object; this attribute is a authentication signature built from my secret key and the full message to be sent:
    oauth_request.sign_request(oauth.OAuthSignatureMethod_HMAC_SHA1(), consumer, token)

    return oauth_request.to_url()


def detect_api_version(url):
    """
    Identify twitter API version from `url` (str).
    Returns a str (e.g.: '1.1' or '2').
    """
    
    # Get start position of the remaining URL after `preface`:
    preface = 'api.twitter.com/'
    url_path = url.find(preface) + len(preface)
    # Extract version:
    version = url[url_path:].split('/')[0]
    
    return version


def get_credentials(credentials):
    """
    Return credentials referenced as input.
    If `credentials` is a dict, return it.
    If it is a str, treat it as a path to a
    JSON file and load it.
    
    Returns a dict.
    """
    # Check if credentials have been provided or if it must be loaded:
    cred_type = type(credentials)
    if cred_type != dict:
        if cred_type == str:
            # Load credentials from JSON file:
            with open(credentials, 'r') as f:
                credentials = json.load(f)
        else:
            raise Exception('Unknown `credentials` type `{}`.'.format(cred_type))
    
    return credentials


def std_parameters(param_value):
    """
    Standardize a HTTP request parameter 
    value to be inserted in the URL. Values
    that do not need standardizing are 
    returned as is.
    """
    # Boolean:
    if type(param_value) is bool:
        param_value = str(param_value).lower()
    # List or tuples:
    if type(param_value) in (list, tuple):
        param_value = list2csv(param_value)
    
    return param_value
       
 
def join_pars_to_url(url, parameters):
    """
    Join `parameters` (dict) to the `url`
    (str) as standard query parameters.
    
    Returns the full URL (str).
    """
    url_options = '&'.join([str(k) + '=' + str(std_parameters(v)) for k,v in parameters.items() if v is not None])
    if len(url_options) > 0:
        final_url = url + '?' + url_options
    else:
        final_url = url
    
    return final_url


def auth_in_header(header, credentials):
    """
    Place twitter API authentication elements
    into the HTTP request header.
    
    Parameters
    ----------
    header : dict
        HTTP request header, either an 
        empty dict or one containing 
        other header elements.
    credentials : dict
        A dict containing the Bearer token.
    
    Returns
    -------
    auth_header : dict
        The input `header` with authentication
        elements added to it.
    """
    
    # For OAuth 2.0 Bearer Token (also known as app-only):
    token = credentials['bearer_token']
    header.update({'authorization': 'Bearer ' + token})
    
    return header


def exceeded_request_limit(response, abort=True):
    """
    Check if the `response` from a twitter 
    API request informs that the request 
    limit has been exceeded. In this case,
    return True. Otherwise, return False.
    If `abort` is True, run `sys.exit()` on 
    exceeding limit.
    """
    
    # Look for rate limit error in the response body:
    body = json.loads(response.text)
    rate_limit = False
    if 'errors' in body:
        if type(body['errors']) is list:
            for err in body['errors']:
                if 'code' in err.keys():
                    if err['code'] == 88:
                        rate_limit = True
        else:
            raise Exception("Expecting a list under body['errors'], but found {}".format(type(body['errors'])))
    
    # Check for status code or error message:
    if response.status_code == 429 or rate_limit:
        print('!! Request limit exceeded. Request header: {}'.format(response.headers))
        if abort is True:
            sys.exit()
        return True
    
    return False


def request_twitter_api(resource_url, parameters=None, header=None, method='GET', return_header=False, credentials='/home/hxavier/config/keys/twitter_api_key.json'):
    """
    Make a request to a Twitter API endpoint and return the response
    as a parsed JSON (i.e. a list of dicts or dict of lists).
    
    Input
    -----
    
    resource_url : str
        The API endpoint URL address. Check API reference under 
        https://developer.twitter.com/en/docs/twitter-api/v1 or
        https://developer.twitter.com/en/docs/api-reference-index
        for available endpoints. Working for GET methods. Maybe POST methods
        require a change in the `oauth` code.
        
    parameter : dict or None
        The paremeters (keys) and their values (values) for the endpoint 
        request. Check the reference mentioned above for available 
        parameters.
    
    header : dict or None
        Header to be passed to HTTP request (e.g. may include 
        authentication tokens).
    
    method : str 
        The HTTP method to use. It can be 'GET' or 'POST' and it 
        should match the type of the endpoint specified in `resource_url`.
    
    return_header : bool
        Whether to also return the header of the request or not.

    credentials: dict or str
        if dict:
            Credentials generated when creating a project and an app
            at https://developer.twitter.com. For v1.1 requests, the 
            necessary ones in this dict are: 'api_key', 'api_secret_key', 
            'access_token' and 'access_token_secret'. For v2 requests,
            this codes uses the 'bearer_token'.
        if str:
            Filename of a JSON file containing the credentials as
            described above.
        
    Return
    ------
    
    data : dict
        The response from the API endpoint.

    headers : dict (if `return_header` is True)
        The header of the HTTP request.
    """

    # Input and sanity checks:
    assert method == 'GET' or method == 'POST', "Unknown HTTP method {}; it should be 'GET' or 'POST'."
    
    # Standardize input:
    if parameters is None:
        parameters = dict()
    if header is None:
        header = dict()
    
    # Load credentials in case of a JSON filename:
    credentials = get_credentials(credentials)

    # Get API version:
    version = detect_api_version(resource_url)

    # Prepare for API version:
    if version == '1.1':
        # Create URL for rest API v1:
        url = augment(resource_url, parameters, credentials, method)
    elif version == '2':
        url = join_pars_to_url(resource_url, parameters)
        header = auth_in_header(header, credentials)
    else:
        raise Exception('Unknown twitter API version {}'.format(version))
    
    # Prompt the API and get the response:
    if method == 'POST':
        response = requests.post(url, headers=header)
    else:
        response = requests.get(url, headers=header)
    # Check if request exceeded rate limit:
    exceeded_request_limit(response)

    # Get content if existent:
    if response.status_code != 200:
        print('!! ERROR !!')
        print('Header:', response.headers)
        print('Body:', response.text)
        raise Exception("Request failed ({}): {}.".format(response.status_code, response.reason))
    else:
        data = json.loads(response.content.decode('utf-8'))

    if return_header:
        return data, response.headers
    else:
        return data
