import numpy as np


class MultiArmedBandit:
    def __init__(self, reward_probability_list):
        """ Create a new bandit environment.

        It is possible to pass the parameter of the simulation.
        @param: reward_probability_list: each value correspond to
            probability [0, 1] of obtaining a positive reward of +1.
            For each value in the list an arm is defined.
            e.g. [0.3, 0.5, 0.2, 0.8] defines 4 arms, the last one
            having higher probability (0.8) of returning a reward.
        """
        self.reward_probability_list = np.array(reward_probability_list)

    def step(self, action):
        """Pull the arm indicated in the 'action' parameter.

        @param: action an integer representing the arm to pull.
        @return: reward it returns the reward obtained pulling that arm
        """
        if action > len(self.reward_probability_list):
            raise Exception("Invalid action!")
        p = self.reward_probability_list[action]
        q = 1.0 - p
        return None, np.random.choice(2, p=[q, p]), False, False, {}

    def reset(self):
        return None, None

    @property
    def state_dim(self):
        return None

    @property
    def action_dim(self):
        return self.reward_probability_list.shape[0]

    @property
    def utility(self):
        return self.reward_probability_list


