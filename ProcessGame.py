import Game
import socket
from MessageParser import GameMessageParser
import sys
from Config import Config

class ProcessGame(object):

    def __init__(self, id):
        self.id = id
        self.game = getattr(Game, Config.GAME_NAME)()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.msg_parser = GameMessageParser()


    def run(self):
        port = Config.WORKER_BASE_PORT + self.id
        self.sock.connect(('localhost', port))

        self.msg_parser.send_game_model_info(self.sock, self.game.get_game_model_info())

        state, reward, done, next_state = self.game.reset()
        while True:
            self.msg_parser.send_sample(self.sock, [state, reward, done, next_state])
            if not done:
                action = self.msg_parser.recv_action(self.sock)
                state, reward, done, next_state = self.game.step(action)
            else:
                state, reward, done, next_state = self.game.reset()


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: python Worker.py int(id)')
        sys.exit(0)

    if not Config.GAME_PUSH_ALGORITHM:
        sys.exit(0)

    game = ProcessGame(int(sys.argv[1]))
    game.run()
