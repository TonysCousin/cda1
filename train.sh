#! /bin/bash

echo "Copying code..."
cd ~/projects/cda1
cp *.py staging
cp -r docs staging

echo "Starting..."
cd staging

# When using fractional GPU over multiple workers, the cuda sw spews out a harmless warning about
# inability to parse CUBLAS_WORKSPACE_CONFIG, which can't be fixed, so we just ignore those messsages with grep.
# The second variant of the command below performs ~3% slower than the first one, but the first one leaves
# all the warning messages in the stored log file, only removing them from stdout.
python -u train.py $1 > >(tee ~/tmp/log) 2> >(tee -a ~/tmp/log) >&2 | grep -v "F.linear" | grep -vi CUBLAS
#python -u train.py $1 > >(tee ~/tmp/log) 2> >(grep -vi CUBLAS | grep -v "F.linear" | tee -a ~/tmp/log) >&2
