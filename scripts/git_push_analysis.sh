#!/usr/bin/env bash

echo "Enter upload script..."
stamp=`date +%Y-%m-%d_%H:%M:%S`
msg="Auto-update webpage data >> $stamp"
cd /home/hxavier/projetos/violentometro/

echo "Add updates..."
git add webpage/data

echo "Commit updates..."
git commit -m "$msg"

echo "Push updates..."
git push
