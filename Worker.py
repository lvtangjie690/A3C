# Copyright (c) 2016, hzlvtangjie. All rights reserved.

import sys, time

from MessageParser import ServerMessageParser, GameMessageParser

from Config import Config
import Const

from NetworkVP import NetworkVP
from Experience import Experience

from threading import Thread
import numpy as np
import queue
import socket

import Game

class MsgQueue(object):

    def __init__(self):
        self.state_queue = queue.Queue(maxsize=1)
        self.action_queue = queue.Queue(maxsize=1)
        self.sample_queue = queue.Queue(maxsize=1)
        
        self.game_model_info_queue = queue.Queue(maxsize=1)

    def get_state(self):
        return self.state_queue.get()

    def step(self, action):
        self.action_queue.put(action)
        sample = self.sample_queue.get()
        return sample
    
    def reset(self):
        pass

    def get_game_model_info(self):
        return self.game_model_info_queue.get()

class ThreadWorkerListener(Thread):

    def __init__(self, id, msg_queue):
        super(ThreadWorkerListener, self).__init__()
        self.setDaemon(True)
        self.id = id
        self.last_action = None

        self.msg_queue = msg_queue
        self.game_msg_parser = GameMessageParser()

    def init_listener(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port = Config.WORKER_BASE_PORT + self.id
        self.server.bind(('localhost', port))
        self.server.listen(1)

    def run(self):
        print('ThreadWorkerListener Start Running')
        self.init_listener()
        while True:
            conn, addr = self.server.accept()
            print('worker %d accept connection'%self.id, conn, addr)
            #recv game's state_size and action_size for model init
            game_model_info = self.game_msg_parser.recv_game_model_info(conn)
            self.msg_queue.game_model_info_queue.put(game_model_info)
             
            while True:
                sample = self.game_msg_parser.recv_sample(conn)
                state, reward, done, next_state = sample
                if self.last_action != None:
                    self.msg_queue.sample_queue.put(sample)
                if not done:
                    self.msg_queue.state_queue.put(next_state)
                    self.last_action = self.msg_queue.action_queue.get()
                    self.game_msg_parser.send_action(conn, self.last_action)
                else:
                    self.last_action = None

class Worker(object):
    def __init__(self, id):
        super(Worker, self).__init__()
        self.id = id

        self.discount_factor = Config.DISCOUNT

        self.device = Config.DEVICE
        self.model = None

        self.server_msg_parser = ServerMessageParser()

        self.master = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.master.connect((Config.MASTER_IP, Config.MASTER_PORT))
        
        if Config.GAME_PUSH_ALGORITHM:
            self.game = MsgQueue()
            self.listener = ThreadWorkerListener(self.id, self.game)
            self.listener.start()
        else:
            self.game = getattr(Game, Config.GAME_NAME)()


    def init_model(self, state_space_size, action_space_size):
        self.num_actions = action_space_size
        self.actions = np.arange(self.num_actions)

        self.model = NetworkVP(self.device, Config.NETWORK_NAME, state_space_size, action_space_size)


    @staticmethod
    def _accumulate_rewards(experiences, discount_factor, terminal_reward, done):
        exp_length = len(experiences) if done else len(experiences)-1
        reward_sum = terminal_reward
        for t in reversed(range(0, exp_length)):
            r = np.clip(experiences[t].reward, Config.REWARD_MIN, Config.REWARD_MAX)
            reward_sum = discount_factor * reward_sum + r
            experiences[t].reward = reward_sum
        return experiences[:exp_length]

    def convert_data(self, experiences):
        x_ = np.array([exp.state for exp in experiences])
        a_ = np.eye(self.num_actions)[np.array([exp.action for exp in experiences])].astype(np.float32)
        r_ = np.array([exp.reward for exp in experiences])
        return x_, r_, a_

    def select_action(self, prediction):
        if Config.PLAY_MODE:
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.actions, p=prediction)
        return action

    def predict_p_and_v(self, state):
        predictions, values = self.model.predict_p_and_v([state,])
        return predictions[0], values[0]

    def run_episode(self):
        self.game.reset()
        done = False
        experiences = []

        time_count = 0
        reward_sum = 0.0

        while not done:
            # very first few frames
            
            prediction, value = self.predict_p_and_v(self.game.get_state())
            action = self.select_action(prediction)
            state, reward, done, next_state = self.game.step(action)
            # state, action, reward, done, next_state
            reward_sum += reward
            exp = Experience(state, action, prediction, reward, done)
            experiences.append(exp)

            if done or time_count == Config.TIME_MAX:
                terminal_reward = 0 if done else value
                updated_exps = Worker._accumulate_rewards(experiences, self.discount_factor, terminal_reward, done)

                x_, r_, a_ = self.convert_data(updated_exps)
                yield x_, r_, a_, reward_sum

                # reset the tmax count
                time_count = 0
                # keep the last experience for the next batch
                experiences = [experiences[-1]]
                reward_sum = 0.0

            time_count += 1

    def run(self):
        print('Worker %d Start Running'%self.id)

        state_space_size, action_space_size = self.game.get_game_model_info()
        self.init_model(state_space_size, action_space_size)
        
        self.server_msg_parser.send_to_master(self.master, Const.MSG_TYPE_INIT, (state_space_size, action_space_size))
        model = self.server_msg_parser.recv_from_master(self.master)
        self.model.update(model)

        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while True:
            total_reward = 0
            total_length = 0
            for x_, r_, a_, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(r_) + 1  # +1 for last frame that we drop
                # compute gradients and send to the master
                gradients = self.model.compute_gradients(x_, r_, a_)
                self.server_msg_parser.send_to_master(self.master, Const.MSG_TYPE_GRADIENTS, gradients)
                model = self.server_msg_parser.recv_from_master(self.master)
                self.model.update(model)
            self.server_msg_parser.send_to_master(self.master, Const.MSG_TYPE_LOG, (total_reward, total_length))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Worker.py id")
        sys.exit(0)
    worker = Worker(int(sys.argv[1]))
    worker.run()
