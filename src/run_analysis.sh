#!/usr/bin/env bash

# Require an argument to run:
if [ "$#" -ne 1 ]; then
    echo "Please inform a suffix for the log file."
    exit
fi

# Get input:
date=$1

# Run:
source ../env/bin/activate
nohup ./tweet_analysis.py > ../tweets/logs/twitter_analysis_$date.log 2>&1 &
