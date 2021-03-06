#!/bin/bash

# start master

pid_file=a3c.pid

workers=16

echo "start master..."
python Master.py $workers &

echo $! >> $pid_file

sleep 10s

# start games
echo "start games..."

for i in $(seq 0 $(($workers-1)))
do
    python ProcessGame.py &
    echo $! >> $pid_file
    sleep 0.1s
done
