#!/bin/bash

# start master
pid_file=a3c.pid

cat $pid_file | xargs kill -9
echo "" > $pid_file

ps -axu|grep Master|grep python|awk '{print $2}'|xargs kill -9 
