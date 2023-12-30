#! /bin/bash

# This script splits the full data file (without header line) into a training set and
# a test set. It does not yet worry about splitting out a verification set.
FULL="full.csv"
TRAIN="train.csv"
#VER="ver.csv"
TEST="test.csv"

FULL_LINES=`wc -l $FULL | awk  '{print $1}'`
TT_LINES=267886 #this is 15% of the number of full data records

# Take 15% off the end of the full file
TRAIN_LINES=$((FULL_LINES-TT_LINES))
echo "FULL_LINES = $FULL_LINES, TT_LINES = $TT_LINES, TRAIN_LINES = $TRAIN_LINES"
echo "Writing training file..."
head -$TRAIN_LINES $FULL > train_raw
cat header train_raw > $TRAIN

# Use that 15% to become the test file
echo "Writing $TT_LINES lines to test file..."
tail -$TT_LINES $FULL > test_raw
cat header test_raw > $TEST

# Clean up
/bin/rm train_raw test_raw
