# Copyright (c) 2016, hzlvtangjie. All rights reserved.

import platform
if platform.python_version()[0] == '2':
    import Queue as queue
else:
    import queue

from threading import Thread, Lock
import socket

from MessageParser import ServerMessageParser
from Config import Config
import Const
from NetworkVP import NetworkVP

from multiprocessing import Queue, Process

import time

init_model_lock = Lock()

class WorkerSocketThread(Thread):
    """Send and recv thread for a worker
    """

    def __init__(self, id, master, sock):
        super(WorkerSocketThread, self).__init__()
        self.id = id
        self.master = master
        self.sock = sock
        self.server_msg_parser = ServerMessageParser()
        self.send_queue = queue.Queue(maxsize=1)

    def run(self):
        while True:
            type, info = self.server_msg_parser.recv_from_worker(self.sock)
            if type == Const.MSG_TYPE_GRADIENTS:
                self.master.gradients_queue.put((self.id, info))
                model = self.send_queue.get()
                self.server_msg_parser.send_to_worker(self.sock, model)
            elif type == Const.MSG_TYPE_INIT:
                init_model_lock.acquire()
                if self.master.model is None:
                    #use worker 0's info to init model
                    self.master.init_model(info[0], info[1])
                init_model_lock.release()
                model = self.master.model.dumps()
                self.server_msg_parser.send_to_worker(self.sock, model)
            elif type == Const.MSG_TYPE_LOG:
                self.master.log_queue.put(info)

class TrainingThread(Thread):
    """Master's training thread
    """

    def __init__(self, master):
        super(TrainingThread, self).__init__()
        self.master = master 

    def run(self):
        print("TrainingThread starts running")
        while True:
            id, gradients = self.master.gradients_queue.get()
            self.master.model.apply_gradients(gradients)
            self.master.workers[id].send_queue.put(self.master.model.dumps())

class Stats(Process):

    MAX_RECENT_RESULTS_SIZE = 100

    def __init__(self, log_queue):
        super(Stats, self).__init__()
        self.log_queue = log_queue
       
        self.episode_count = 0 
        self.recent_reward_results = []
        self.frame_count = 0

        self.last_frame_count = 0
        self.last_time = 0
        self.last_fps = 0

    def run(self):
        print("Stats starts running")
        self.last_time = time.time()
        while True:
            total_reward, total_length = self.log_queue.get()
            self.episode_count += 1
            self.frame_count += total_length
            self.recent_reward_results.append(total_reward)
            while len(self.recent_reward_results) > self.MAX_RECENT_RESULTS_SIZE:
                self.recent_reward_results.pop(0)
            recent_avg_reward = sum(self.recent_reward_results)/float(len(self.recent_reward_results))
            if self.episode_count % 50 == 0:
                if time.time() - self.last_time > 10:
                    self.last_fps = int((self.frame_count - self.last_frame_count)/(time.time() - self.last_time))
                    self.last_frame_count = self.frame_count
                    self.last_time = time.time()
                elif self.last_fps == 0:
                    self.last_fps = int(self.frame_count/(time.time() - self.last_time))
                print(
                    'Episode %d: Recent Avg Reward %.2f, FPS %d'\
                    %(self.episode_count, recent_avg_reward, self.last_fps)
                    )


class Master(object):

    def __init__(self):
        self.workers = []
        device = 'cpu:0'
        self.device = device
        self.model = None
        self.gradients_queue = queue.Queue(maxsize=100)

        self.log_queue = Queue(maxsize=100)
        self.stats = Stats(self.log_queue) 

        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((Config.MASTER_IP, Config.MASTER_PORT))
        self.server.listen(32)

    def init_model(self, state_space_size, action_space_size):
        self.model = NetworkVP(self.device, Config.NETWORK_NAME, state_space_size, action_space_size)


    def add_worker(self, sock):
        worker = WorkerSocketThread(len(self.workers), self, sock)
        self.workers.append(worker)
        self.workers[-1].start()

    def remove_workers(self):
        while len(self.workers) > 0:
            worker_thread = self.workers.pop(0)
            worker_thread.join()

    def run(self):
        self.stats.start()

        self.training_thread = TrainingThread(self)
        self.training_thread.start()

        while True:
            conn, addr = self.server.accept()
            print('Master accept connection', conn, addr)
            self.add_worker(conn)
            time.sleep(0.1)

        # remove worker thread
        self.remove_workers()
        # close training thread
        self.training_thread.join()
        # close stats process 
        self.stats.join()


if __name__ == "__main__":
    master = Master()
    master.run()
