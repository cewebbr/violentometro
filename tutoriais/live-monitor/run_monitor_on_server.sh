#!/usr/bin/env bash

# Require an argument to run:
if [ "$#" -ne 1 ]; then
    echo "Please inform a suffix for the log file."
    exit
fi

# Get input:
date=$1

# Run stuff:
#source ../../env/bin/activate
eval "$(conda shell.bash hook)"
conda activate tf

# Tweet capture:
nohup ../../src/tweet_capture.py config/tweet_capture_config.json > tweets/logs/twitter_capture_$date.log 2>&1 &
# Data analysis:
nohup ../../src/tweet_analysis.py config/tweets2metric_config.json > tweets/logs/twitter_analysis_$date.log 2>&1 &
# Tweet rating:
nohup ../../src/tweet_rating.py config/tweet_capture_config.json > tweets/logs/twitter_rate_$date.log 2>&1 &
