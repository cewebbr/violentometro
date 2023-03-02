# Live violence monitor

This tutorial shows how to setup a system to monitor the level of violence directed to a set of Twitter profiles on the hour.


## Overview

The monitor consists of three processes (Python scripts) that should run permanently and in parallel:

1. capturing tweets using the Twitter API;
2. applying an AI model to the captured tweets to estimate the violence level of the tweets;
3. analysing the collected and scored data in an aggregated way to produce general violence metrics.

The python scripts are ran periodically with intervals given by JSON config files. The last script also pushes the new metrics
to a gitpage.


## Prerequisites


### Tensorflow and other python packages

This project requires tensorflow for rating the tweets and works best when using GPU. To install Tensorflow with GPU support, follow the
instructions in this link: <https://www.tensorflow.org/install/pip>. Once installed, it is recommended to install the remaining necessary
packages for the project in the same environment (Tensorflow recommends using miniconda) with pip. For that, go to the root folder of
this project and run (assuming you have pip installed):

    conda activate tf
    pip install -r requirements.txt
    conda deactivate


### Twitter API bearer token

The Python code employed in this tutorial makes use of the Twitter API v2 ("mentions" endpoint), which can only be accessed using a private token
(the bearer token) associated to a Twitter developer account. Follow the instructions below to get one for yourself.

1. Create a twitter account at: <https://twitter.com/i/flow/signup>.
2. Become a developer at: <https://developer.twitter.com/en/portal/petition/essential/basic-info>.
3. Create a new project by following the instructions at: <https://developer.twitter.com/en/docs/projects/overview>.
4. Follow the same instructions above to create an App for your project.
5. When you create your Twitter App, you will be presented with your API Key and Secret, along with a Bearer Token. You should save these locally as they are only showed once.

You should save the tokens in the file at [../../keys/twitter_violentometro.json](../../keys/twitter_violentometro.json), in the following format:

    {
        "api_key": "<your twitter API key>",
        "api_secret_key": "<your twitter API secret key>",
        "bearer_token": "<your twitter API bearer key>",
        "access_token": "<your twitter API access token>",
        "access_token_secret": "<your twitter API token secret>"
    }

For this project, you only need to put your Bearer token (also known as app-only token) in the JSON above.


### Token access to github

To allow the script to automatically push the new metrics to a [gitpage](), it is necessary to set up a passwordless push with ssh keys.
Follow this instructions to set github in such a way:
<https://stackoverflow.com/questions/8588768/how-do-i-avoid-the-specification-of-the-username-and-password-at-every-git-push>


## Running the monitor


### Setting up

The monitor settings are stored in two JSON files in the [config](config) folder:
[tweet_capture_config.json](config/tweet_capture_config.json), used by the capturing and the rating scripts; and
[tweets2metric_config.json](config/tweets2metric_config.json), used by the analysis script.
We describe the entries in these files below.

#### tweet_capture_config.json

`curr_level`: current number of tweets captured within the monthly cap period;
`cap_renew_date`: date (in format YYYY-MM-DD) when the monthly cap is renewed;
`avg_tweets_per_cand`: average number of tweets to be captured per capturing period; 
`tweets_buffer`: number of tweets NOT to capture in order to avoid exceeding the monthly quota (e.g. 100);
`capture_period`: interval between captures (also the time interval that a capture should cover), in hours;
`tweet_cap`: monthly cap on the number of tweets allowed by the Twitter API (e.g. 2000000 for extended usage);
`max_batch_cands`: maximum number of profiles to collect mentions to in one capture (I suggest 1/3 of the total number of profiles);
`log_dir`: `tweets/logs/capture/`, where capture logs are saved;
`error_dir`: `tweets/logs/capture_errors/`, where error messages during capture are saved;
`data_dir`: `tweets/data/`, where the captured tweets are saved;
`batch_ref_time`: a datetime in the past from which to calculate the next capture instants (format YYYY-MM-DD HH:MM:SS);
`twitter_ids_file`: `config/target_twitter_profiles.csv`, List of Twitter profiles to check for mentions to;
`scored_dir`: `tweets/scored/`, where to save the tweets rated with respect to their violence levels;
`model_dir`: `../../modelos/bertimbau-hatespeech-v01`, folder of the AI model to be used for identifying violence;
`force_rate`: whether the rating script should only rate yet unrated tweets (`false` unless you want to redo all ratings).

#### tweets2metric_config.json

`analyse_ref_time`: a datetime in the past from which to calculate the next analysis instants;
`analyse_period`: time interval between the analysis (and aggregation time interval), in hours;
`id_pool_sources`: `config/twitter_id_pool_sources.log`, the list of lists of twitter profile IDs to be monitored, in case these changed during the monitoring the process;
`id_pool_dir`: `config/`, the folder where the lists of profile IDs are saved;
`batch_logs_dir`: `tweets/logs/capture/`, where the logs of the captures are saved;
`scored_tweets_dir`: `tweets/scored/`, where the scored tweets are saved;
`target_model`: `../../modelos/nb_wrong_target_classifier_v02/production_model.joblib`, a model that identifies if the comment in the tweet refers to the mentioned profile or not;
`bad_users`: list of user IDs in the list of profiles to ignore;
`pool_factor_file`: `config/id_pool_weights.csv`, table informing how to convert the estimated tweet and violent tweet counts from one list of profile IDs to another;
`cand_sel_query`: Pandas query to select a subset of profile IDs to capture mentions to;
`capture_period`: interval between captures (also the time interval that a capture should cover), in hours;
`official_start`: when the analysis officialy started (plot start, format YYYY-MM-DD HH:MM:SS);
`official_end`: when the analysis officialy started (plot start, format YYYY-MM-DD HH:MM:SS);
`time_series_plot`: `../../assets/img/graficos/tweets_agressivos_por_dia_feminino.png`, where to save the time series plot;
`time_series_csv`:  `../../assets/data/tweets_agressivos_por_dia_feminino.csv`, where to save the metrics data;
`webpage_json_file`: `../../assets/js/data.js`, where to save the "big numbers" data.


### Running

On Unix-like systems (e.g. Linux and Mac) and when logged on a server through ssh, you may run the monitoring system
by executing the command `./run_monitor_on_server.sh <run tag>` in this folder, where `<run tag>` is replaced by some
identifier of the current monitor run, like the date (e.g. 2023-03-01). This shell script does three things:

1. Starts the `tf` conda environment;
2. Starts 3 independent processes (tweet capture, tweet rating for violence and data analysis) with [nohup](https://en.wikipedia.org/wiki/Nohup); and
3. Redirect their outputs to logs located at [tweets/logs/](tweets/logs/), identified by the `<run tag>`.

With nohup, you may logout from the server while the three processes responsible for the violence monitoring will stay running.
