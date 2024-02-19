from typing import Union, Optional
from copy import copy

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete
from rich import print

BOARD_STRING = """
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 ? | ? | ? | ? | ? | ? | ?
---|---|---|---|---|---|---
 1   2   3   4   5   6   7
"""


class Connect4Env(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    def __init__(self, render_mode: Optional[str] = None):
        # 6x7 board
        # use the following coordinate system:
        #   ^
        #   |
        #   |
        #   |
        # 0 +-------->
        #   0
        self.board = np.zeros((6, 7))
        self.player = 1
        self.done = False

        self.action_space = Discrete(7)
        self.observation_space = Discrete(7)

    def step(self, action):
        col = action
        row = np.argmax(self.board[:, col] == 0)

        invalid_move = self.board[row, col] != 0

        if not self.done and not invalid_move:
            self.board[row, col] = self.player

        # The reward is computed as follows:
        # * 0 if the game is **already** over. This is to ignore nodes below terminal nodes.
        # * -1 if the move is invalid
        # * 1 if the move won the game for the current player
        # * 0 if the move caused a draw
        # * (impossible for Connect 4) -1 if the move lost the game for the current player
        reward = 0.0
        if self.done:
            reward = 0.0
        elif invalid_move:
            reward = -1.0
        else:
            reward = self._get_winner_state() * self.player

        # We end the game if:
        # * the game was already over
        # * the move won or lost the game
        # * the move was invalid
        # * the board is full (draw)
        done = self.done or reward != 0 or invalid_move or np.all(self.board[-1] != 0)

        if not done:  # Switch roles
            self.player = -self.player

        return self.board, reward, done, False, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        super(Connect4Env, self).reset()
        self.board = np.zeros((6, 7))
        self.player = 1
        self.done = False

        return self.board, {}

    def render(self):
        board_str = copy(BOARD_STRING)
        for i in reversed(range(self.board.shape[0])):
            for j in range(self.board.shape[1]):
                board_str = board_str.replace('?', '[green]X[/green]' if self.board[i, j] == 1 else '[red]O[/red]' if self.board[i, j] == -1 else ' ', 1)
        print(board_str)

    def action_mask(self):
        return self.board[-1] == 0

    def get_state(self):
        return dict(
            board=copy(self.board),
            player=copy(self.player),
            done=copy(self.done),
            valid_mask=copy(self.action_mask()),
            valid_moves=np.arange(self.action_space.n)[self.action_mask()]
        )

    def set_state(self, state):
        self.board = copy(state["board"])
        self.player = copy(state["player"])
        self.done = copy(state["done"])

    def _get_winner_state(self):
        def horizontals(board: np.ndarray):
            return np.stack([
                board[i, j:j + 4]
                for i in range(board.shape[0])
                for j in range(board.shape[1] - 3)
            ])

        def verticals(board: np.ndarray):
            return np.stack([
                board[i:i + 4, j]
                for i in range(board.shape[0] - 3)
                for j in range(board.shape[1])
            ])

        def diagonals(board: np.ndarray):
            return np.stack([
                np.diag(board[i:i + 4, j:j + 4])
                for i in range(board.shape[0] - 3)
                for j in range(board.shape[1] - 3)
            ])

        def antidiagonals(board: np.ndarray):
            return np.stack([
                np.diag(board[i:i + 4, j:j + 4][::-1])
                for i in range(board.shape[0] - 3)
                for j in range(board.shape[1] - 3)
            ])

        all_lines = np.concatenate((
            horizontals(self.board),
            verticals(self.board),
            diagonals(self.board),
            antidiagonals(self.board),
        ))
        # x_won and o_won are 1 if the player won, 0 otherwise
        x_won = np.any(np.all(all_lines == 1, axis=1)).astype(np.int8)
        o_won = np.any(np.all(all_lines == -1, axis=1)).astype(np.int8)

        # We consider the following cases:
        # - !x_won and !o_won -> 0 - 0 = 0 -> draw OR not finished
        # - x_won and !o_won -> 1 - 0 = 1 -> Player 1 (X) won
        # - !x_won and o_won -> 0 - 1 = -1 -> Player -1 (O) won
        # - x_won and o_won -> impossible, the game would have ended earlier
        return x_won - o_won


if __name__ == "__main__":
    env = Connect4Env()

    env.reset()  # Random actions
    done = False
    while not done:
        action = env.action_space.sample()
        obs, rew, done, trunc, info = env.step(action)
        print("Valid moves:", env.action_mask())
        env.render()

    env.reset()  # Test reset-ability
    state = env.get_state()
    actions = np.arange(env.action_space.n)
    for action in actions:
        env.step(action)
        env.render()
        env.set_state(state)
