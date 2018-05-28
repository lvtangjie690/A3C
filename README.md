# Instruction


* A3C Algorithms
    1. Support multi-process architecture
    2. Need install tensorflow and numpy
    3. Only support python3
    4. It will start 1 master and n workers when running

* Support Windows and Linux

* Two Modes:  Algorithm Push Mode & Game Push Mode. 
    
    * if you set Config.GAME_PUSH_ALGORITHM = True, then the program will be running in Game Push Mode, otherwise, Algorithm Push Mode.

# Usage:

* Linux:
    * start: sh start.sh 
    * stop:  sh stop.sh

* Windows: You should start the python program by yourself
    * start: 
        1. start the master:  python Master.py
        2. start the game:    python ProcessGame.py
    * stop: 
        1. kill the running processes