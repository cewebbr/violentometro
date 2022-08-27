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
nohup ./tweet_capture.py > ../tweets/logs/twitter_run_$date.log 2>&1 &
