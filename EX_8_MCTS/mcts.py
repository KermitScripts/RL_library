from abc import ABC, abstractmethod
from typing import Any, Dict
from copy import deepcopy
import random

import numpy as np
from tqdm import tqdm

from env import Connect4Env


def random_valid_action(env: Connect4Env):
    """Returns a random valid action for the current environment."""
    actions = np.arange(env.action_space.n)
    valid_actions = actions[env.action_mask()]
    return np.random.choice(valid_actions)


def randomized_argmax(x: np.ndarray):
    """Returns a random index from all the indices that have max value."""
    return np.random.choice(np.argwhere(x == x.max()).flatten())


class MCTSNode:
    def __init__(
            self,
            env: Connect4Env,
            state: Dict[str, Any],
            c_param: float = np.sqrt(2),
            action: int = None,  
            reward: float = None,
            done: bool = None,
            parent: "MCTSNode" = None
    ):
        self.env = deepcopy(env)  
        self.state = state
        self.c_param = c_param

        self.action = action
        self.reward = reward
        self.done = done
        self.parent = parent
        self.expanded = False

        self.num_actions = self.env.action_space.n

        self.valid_mask = self.state["valid_mask"] 
        self.valid_moves = self.state["valid_moves"] 
        self.player = self.state["player"]

        self.child_num_visits = np.zeros(self.num_actions)
        self.child_value_sum = np.zeros(self.num_actions)
        self.children = [None] * self.num_actions

    @property
    def value(self):
        """Returns the value of this node."""
        return self.parent.child_value_sum[self.action]

    @value.setter
    def value(self, x):
        """Sets the value of this node."""
        self.parent.child_value_sum[self.action] = x

    @property
    def num_visit(self):
        """Return the visit counter for this state."""
        return self.parent.child_num_visits[self.action]

    @num_visit.setter
    def num_visit(self, x):
        """Sets the visit counter for this state."""
        self.parent.child_num_visits[self.action] = x

    def child_values(self):
        """Returns all the child values (i.e. value sum divided by visit counts).
        Take into special account the children that have not been visited yet (i.e. danger of division by zero).
        """
        values = np.zeros(self.num_actions)
        mask = self.child_num_visits != 0.0
        values[mask] = self.child_value_sum[mask] / self.child_num_visits[mask]
        return values

    def child_exploration(self):
        """Returns the exploration term (weighted by `c_param`).
        Take into special account the children that have not been visited yet (i.e. danger of division by zero).
        """
        num_visits = self.child_num_visits.sum()
        return np.zeros(self.num_actions) if num_visits == 0.0 else self.c_param * np.sqrt(num_visits / (self.child_num_visits + 1.0))

    def uct(self):
        """Computes the UCT terms for all children."""
        return self.child_values() + self.child_exploration()

    def uct_action(self):
        """Samples the next action, based on current UCT values.
        Mask out the UCT values for invalid moves, i.e., by setting them to `-np.inf`. Use the `self.valid_mask` for this.
        """
        uct = self.uct()
        uct[~self.valid_mask] = -np.inf
        return randomized_argmax(uct)

    def selection(self) -> "MCTSNode":
        """Traverses the tree starting from the current node until a not expanded node is reached."""
        current = self
        while current.expanded:
            selected_action = current.uct_action()
            current = current.children[selected_action]
        return current

    def expansion(self) -> "MCTSNode":
        """Expands the node, i.e.:
        1) Create a `MCTSNode` object for every valid move.
        2) Take a random node child and return it.
        """
        if self.done:
            return self

        for action in self.valid_moves:
            self.env.set_state(self.state)
            _, reward, done, _, _ = self.env.step(action)
            state = self.env.get_state()
            node = MCTSNode(self.env, state, self.c_param, action, reward, done, self)
            self.children[action] = node

        self.env.set_state(self.state)
        action = random_valid_action(self.env)
        child_node = self.children[action]
        self.expanded = True
        return child_node

    def simulation(self, num_rollouts_per_simulation: int = 1) -> float:
        """Simulates a `num_rollouts_per_simulation` rollouts from this node.
        Returns the mean value.
        """
        if self.done:
            return self.reward

        reward_sum = 0.0
        for _ in range(num_rollouts_per_simulation):
            self.env.set_state(self.state)
            done = False
            while not done:
                action = random_valid_action(self.env)
                _, reward, done, _, _ = self.env.step(action)
            reward_sum += reward

        return reward_sum / num_rollouts_per_simulation

    def backpropagation(self, reward) -> None:
        """Back propagates the games' outcome up the search tree.
        At every step, you have to take into account which player's turn it was.
        A positive outcome for the player is a negative outcome for the opponent (hint: Flip the reward)
        """
        current = self
        if reward == 0.0:  # We have a draw
            reward = 0.5

        while current.parent is not None:
            current.value = current.value + reward
            current.num_visit = current.num_visit + 1
            reward = 1.0 - reward
            current = current.parent


class BaseAgent(ABC):
    @abstractmethod
    def compute_action(self, env: Connect4Env) -> int:
        ...


class RandomAgent(BaseAgent):
    def compute_action(self, env: Connect4Env) -> int:
        return random_valid_action(env)


class MCTSAgent(BaseAgent):
    def __init__(
            self,
            num_simulations: int,
            num_rollouts_per_simulation: int,
            c_param: float = np.sqrt(2)
    ):
        self.num_simulations = num_simulations
        self.num_rollouts_per_simulations = num_rollouts_per_simulation
        self.c_param = c_param

    def compute_action(self, env: Connect4Env) -> int:
        env = deepcopy(env)  # So that we don't alter the original environment
        root_node = MCTSNode(env, env.get_state(), c_param=self.c_param)  # Create root node

        for _ in range(self.num_simulations):  # Do N simulations
            leaf_node = root_node.selection()
            child_node = leaf_node.expansion()
            terminal_reward = child_node.simulation(self.num_rollouts_per_simulations)
            child_node.backpropagation(terminal_reward)

        child_values = root_node.child_value_sum / root_node.child_num_visits
        return randomized_argmax(root_node.child_value_sum / root_node.child_num_visits)  # Select the action based on some criteria


def arena(player: BaseAgent, opponent: BaseAgent, env: Connect4Env, render: bool = False) -> int:
    env.reset()

    while True:
        player_action = player.compute_action(env)  # Calculate action
        _, reward, done, _, _ = env.step(player_action)  # Query player for action

        if render:  # Render
            env.render()

        # Return will be:
        # * 1 if player won
        # * 0 if we have a draw
        # * -1 if the player did an incorrect move -> opponent won
        if done:  # Check if done, report outcome if so
            return reward

        opponent_action = opponent.compute_action(env)
        _, reward, done, _, _ = env.step(opponent_action)

        if render:
            env.render()

        # Return will be:
        # * -1 if opponent won
        # * 0 if we have a draw
        # * 1 if the player did an incorrect move -> player won
        if done:
            return -reward


if __name__ == '__main__':
    env = Connect4Env()
    obs, _ = env.reset()

    # Initialize the two players
    player = MCTSAgent(
        num_simulations=200,
        num_rollouts_per_simulation=1,
        c_param=1
    )
    opponent = RandomAgent()
    #opponent = MCTSAgent(
    #    num_simulations=200,
    #    num_rollouts_per_simulation=1,
    #    c_param=1
    #)

    # Quantitative evaluation
    eval = False  # Just set this to `False` to skip the quantitative evaluation
    if eval:
        num_games = 10
        player_wins = 0
        opponent_wins = 0
        draws = 0
        for _ in tqdm(range(num_games)):
            outcome = arena(player, opponent, env)
            if outcome == 1:
                player_wins += 1
            elif outcome == -1:
                opponent_wins += 1
            else:
                draws += 1

        print(f"Outcome distribution of {num_games} games player:")
        print(f"Player wins: {player_wins}")
        print(f"Opponent wins: {opponent_wins}")
        print(f"Draws: {draws}")

    # Qualitative evaluation
    arena(player, opponent, env, render=True)


