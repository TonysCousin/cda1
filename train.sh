#! /bin/bash

echo "Copying code..."
cd ~/projects/cda1
cp *.py staging
cp -r docs staging

echo "Starting..."
cd staging
python -u train.py $1 > >(tee ~/tmp/log) 2> >(tee -a ~/tmp/log) >&2
