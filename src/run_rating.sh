#!/usr/bin/env bash

# Require an argument to run:
if [ "$#" -ne 1 ]; then
    echo "Please inform a suffix for the log file."
    exit
fi

# Get input:
date=$1

# Run:
eval "$(conda shell.bash hook)"
conda activate tf
nohup ./tweet_rating.py > ../tweets/logs/twitter_rate_$date.log 2>&1 &
