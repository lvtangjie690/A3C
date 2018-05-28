# Copyright (c) 2016, hzlvtangjie. All rights reserved.

import sys, time

from MessageParser import MessageParser, GameMessageParser

from Config import Config

from NetworkVP import NetworkVP
from Experience import Experience

from threading import Thread
import numpy as np
import queue
import socket

from multiprocessing import Process, Queue

import Game

class FakeGame(object):
    """ FakeGame: recv and send real game's msg and \
        step as a game for worker
    """

    def __init__(self, id):
        self.id = id
        self.state_queue = queue.Queue(maxsize=1)

        self.last_action = None        
        self.game_msg_parser = GameMessageParser()
        #init game listener
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port = Config.WORKER_BASE_PORT + self.id
        self.server.bind(('localhost', port))
        self.server.listen(1)

    def run(self):
        while True:
            # create connection between worker and game
            self.sock, addr = self.server.accept()
            break

    def get_state(self):
        if self.state_queue.empty():
            # the first frame in an episode
            sample = self.game_msg_parser.recv_sample(self.sock)
            state, reward, done, next_state = sample
            return next_state
        else:
            return self.state_queue.get()            


    def step(self, action):
        # send action
        self.game_msg_parser.send_action(self.sock, action)
        # recv new sample
        sample = self.game_msg_parser.recv_sample(self.sock) 
        state, reward, done, next_state = sample
        if not done:
            self.state_queue.put(next_state)
        return sample
    
    def reset(self):
        pass

    def get_game_model_info(self):
        return self.game_msg_parser.recv_game_model_info(self.sock)

def gradients_to_list(gradients):
    gradients = [tuple([var.tolist() for var in item]) for item in gradients]
    return gradients

def list_to_model(model):
    model = [np.array(var, dtype=np.float32) for var in model]
    return model

class Worker(Process):
    def __init__(self, id, master):
        super(Worker, self).__init__()
        self.id = id

        self.discount_factor = Config.DISCOUNT

        self.device = Config.DEVICE
        self.model = None

        self.master = master
        self.model_queue = Queue(maxsize=100)
        
        if Config.GAME_PUSH_ALGORITHM:
            self.game = FakeGame(self.id)
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
        if Config.GAME_PUSH_ALGORITHM:
            self.game.run()

        state_space_size, action_space_size = self.game.get_game_model_info()
        self.init_model(state_space_size, action_space_size)

        self.master.init_queue.put((self.id, state_space_size, action_space_size))
        model = list_to_model(self.model_queue.get())
        self.model.update(model)

        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while True:
            total_reward = 0
            total_length = 0
            for x_, r_, a_, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(r_)
                # compute gradients and send to the master
                gradients = gradients_to_list(self.model.compute_gradients(x_, r_, a_))
                self.master.gradients_queue.put((self.id, gradients))
                # recv model from master
                model = list_to_model(self.model_queue.get())
                self.model.update(model)
            # send log to master
            self.master.log_queue.put((total_reward, total_length))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python Worker.py id")
        sys.exit(0)
    worker = Worker(int(sys.argv[1]))
    worker.run()
