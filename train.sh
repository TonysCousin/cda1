#! /bin/bash

cd ~/projects/cda1
cp *.py staging
cp -r docs staging

cd staging
python -u tune.py > >(tee ~/tmp/log) 2> >(tee -a ~/tmp/log) >&2
