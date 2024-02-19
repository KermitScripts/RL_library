import numpy as np

from gridworld import GridWorldEnv
from util import value_function_policy_plot, estimate_policy_array_from_samples


class TDAgent:
    def __init__(self, env, discount_factor, learning_rate):
        self.g = discount_factor
        self.lr = learning_rate

        self.grid_height = int(env.observation_space.high[0])
        self.grid_width = int(env.observation_space.high[1])

        self.num_actions = env.action_space.n

        self.V = np.zeros([self.grid_height, self.grid_width], dtype=np.float32)
        self.policy = np.ones([self.grid_height, self.grid_width, self.num_actions],
                              dtype=np.float32) / self.num_actions

        self.policy[1, 1] = 0
        self.policy[1, 1, 2] = 1  # Down with 100%

        self.policy[2, 1] = 0
        self.policy[2, 1, 2] = 1  # Down with 100%

        self.policy[1, 0] = 0
        self.policy[1, 0, 2] = 1  # Down with 100%

        self.policy[2, 0] = 0
        self.policy[2, 0, 2] = 1  # Down with 100%

        self.env = env

    def action(self, s):
        # Samples an action based on a state
        action = np.random.choice(np.arange(self.num_actions), p=self.policy[s[0], s[1]])

        return action

    def learn(self, n_timesteps=50000):
        s, info = env.reset()
        s_ = None

        for i in range(n_timesteps):
            a = self.action(s)   
            s_, r, terminated, _, _ = env.step(a)   

            self.update(s, a, r, s_)  

            s = s_

            if terminated:
                s, _ = env.reset()

    def update(self, s, a, r, s_):
        # V(s) <- V(s) + alpha * (r + gamma * V(s') - V(s))
        self.V[s[0], s[1]] = self.V[s[0], s[1]] + self.lr * (r + self.g * self.V[s_[0], s_[1]] - self.V[s[0], s[1]])


if __name__ == "__main__":
    env = GridWorldEnv()
    td_agent = TDAgent(env, discount_factor=0.8, learning_rate=0.01)

    td_agent.learn(n_timesteps=50000)

    V = td_agent.V.copy()
    policy = td_agent.policy
    env_map = td_agent.env.map
    value_function_policy_plot(V, policy, env_map)
