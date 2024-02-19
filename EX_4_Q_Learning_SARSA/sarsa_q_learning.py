import numpy as np
from helper import action_value_plot, test_agent

from gym_gridworld import GridWorldEnv


class SARSAQBaseAgent:
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        self.g = discount_factor
        self.lr = learning_rate
        self.eps = epsilon
        self.env = env

        self.grid_height = int(env.observation_space.high[0])
        self.grid_width = int(env.observation_space.high[1])
        self.num_actions = env.action_space.n
        self.Q = np.zeros([self.grid_height, self.grid_width, self.num_actions], dtype=np.float32)

    def action(self, s, epsilon=None):

        eps = self.eps
        if epsilon is not None:
            eps = epsilon
        if eps >= np.random.uniform(0, 1):
            action = np.random.randint(self.num_actions)
        else:
            max_q_actions = np.argwhere(self.Q[s[0], s[1]] == np.amax(self.Q[s[0], s[1]])).flatten()
            action = np.random.choice(max_q_actions)
        return action


class SARSAAgent(SARSAQBaseAgent):
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        super(SARSAAgent, self).__init__(env, discount_factor, learning_rate, epsilon)

    def learn(self, n_timesteps=200000):

        t = 0
        while t <= n_timesteps:
            terminated = False
            s, _ = self.env.reset()
            a = self.action(s)

            while not terminated:
                s_, r, terminated, _, _ = self.env.step(a)
                a_ = self.action(s_)

                self.update_Q(s, a, r, s_, a_)

                s = s_
                a = a_
                t += 1

    def update_Q(self, s, a, r, s_, a_):

        self.Q[s[0], s[1], a] = self.Q[s[0], s[1], a] + self.lr * (
                    r + self.g * self.Q[s_[0], s_[1], a_] - self.Q[s[0], s[1], a])


class QLearningAgent(SARSAQBaseAgent):
    def __init__(self, env, discount_factor, learning_rate, epsilon):
        super(QLearningAgent, self).__init__(env, discount_factor, learning_rate, epsilon)

    def learn(self, n_timesteps=200000):

        t = 0
        while t <= n_timesteps:
            terminated = False
            s, _ = self.env.reset()

            while not terminated:
                a = self.action(s)
                s_, r, terminated, _, _ = self.env.step(a)

                self.update_Q(s, a, r, s_)

                s = s_
                t += 1

    def update_Q(self, s, a, r, s_):

        self.Q[s[0], s[1], a] = self.Q[s[0], s[1], a] + self.lr * (
                    r + self.g * np.max(self.Q[s_[0], s_[1]]) - self.Q[s[0], s[1], a])


def run():
    env = GridWorldEnv(map_name='standard')
    # env = GridWorldEnv(map_name='cliffwalking')

    discount_factor = 0.9
    learning_rate = 0.05
    epsilon = 0.4
    n_timesteps = 200000

    sarsa_agent = SARSAAgent(env, discount_factor, learning_rate, epsilon)
    sarsa_agent.learn(n_timesteps=n_timesteps)
    action_value_plot(sarsa_agent)
    print('Testing SARSA agent')
    test_agent(sarsa_agent, env, epsilon=0.1)

    qlearning_agent = QLearningAgent(env, discount_factor, learning_rate, epsilon)
    qlearning_agent.learn(n_timesteps=n_timesteps)
    action_value_plot(qlearning_agent)
    print('Testing Q-Learning agent')
    test_agent(qlearning_agent, env, epsilon=0.1)


if __name__ == "__main__":
    run()
