import numpy as np
from envs import MultiArmedBandit
import matplotlib.pyplot as plt


class BaseAgent:
    def __init__(self, env, value_init=1e-8):
        self._env = env
        self._num_arms = env.action_dim
        self._state_dim = env.state_dim
        self._state_action_dim = np.concatenate((env.state_dim, env.action_dim), axis=None) if env.state_dim is not None else env.action_dim
        self._value_estimate = np.full(self._state_action_dim, value_init)
        self._reward_counter = np.zeros(self._state_action_dim)
        self._action_counter = np.ones(self._state_action_dim)

    def train(self, num_epochs=1000, time_steps_per_epoch=1000, render=False, verbose=False):
        epoch_mean_rewards = []
        epoch_utility_estimation_rmse = []

        state, _ = self._env.reset()
        episode_reward = 0.0
        max_reward = -13371337
        for epoch in range(num_epochs):
            epoch_reward = 0.0
            for i in range(time_steps_per_epoch):
                if render:
                    self._env.render()

                action = self.action_selection(state)
                state_action_index = tuple(np.concatenate((state, action), axis=None)) if state is not None else action
                next_state, reward, done, _, info = self._env.step(action)
                self._reward_counter[state_action_index] += reward
                self._action_counter[state_action_index] += 1.0
                # Update the value estimate
                self.update_value_estimate(state, action, reward, next_state, done)

                epoch_reward += reward
                episode_reward += reward
                state = next_state
                if done:
                    if verbose and max_reward < episode_reward:
                        max_reward = max(episode_reward, max_reward)
                        print(f"Timestep {epoch * time_steps_per_epoch + i, episode_reward}, episode reward {episode_reward}, max reward {max_reward}")
                    episode_reward = 0.0
                    state, _ = self._env.reset()

            epoch_mean_rewards.append(epoch_reward / time_steps_per_epoch)
            if hasattr(self._env, 'utility'):
                # This is only applicable for the MAB environment
                utility_estimation_error = np.linalg.norm(self._value_estimate - self._env.utility)
                epoch_utility_estimation_rmse.append(utility_estimation_error)

        return epoch_mean_rewards, epoch_utility_estimation_rmse

    def update_value_estimate(self, state, action, reward, next_state, done):
        state_action_index = tuple(np.concatenate((state, action), axis=None)) if state is not None else action
        # In context-free bandits, the value estimate of a state-action pair is:
        # - the reward received when performing action a in state s
        # - divided by the number n the action a was performed
        self._value_estimate[state_action_index] = self._reward_counter[state_action_index] / self._action_counter[state_action_index]

    def action_selection(self, state):
        return [], []


class RandomAgent(BaseAgent):
    def action_selection(self, state):
        action = np.random.randint(low=0, high=self._num_arms)
        return action


class EpsilonGreedyAgent(BaseAgent):
    def __init__(self, env, epsilon=0.1):
        super().__init__(env)
        self._epsilon = epsilon

    def action_selection(self, state):
        value_estimate = self._value_estimate if state is None else self._value_estimate[tuple(state)]
        if np.random.rand() < self._epsilon:
            return np.random.choice(self._num_arms)

        amax = np.amax(value_estimate)
        indices = np.where(value_estimate == amax)[0]
        action = np.random.choice(indices)
        return action


class UCB1Agent(BaseAgent):
    def __init__(self, env, value_init=1e-8, c=np.sqrt(2)):
        super().__init__(env, value_init=value_init)
        self._c = c

    def action_selection(self, state):
        value_estimate = self._value_estimate if state is None else self._value_estimate[tuple(state)]
        action_counter = self._action_counter if state is None else self._action_counter[tuple(state)]
        timestep = sum(action_counter)
        ucb = value_estimate + self._c * np.sqrt(2 * np.log(timestep) / action_counter)
        action = np.argmax(ucb)
        return action


class ProbabilityMatching(BaseAgent):
    def __init__(self, env):
        super().__init__(env)

    def action_selection(self, state):
        # Remember: The environment always returns a reward from the set {0, 1}
        reward_counter = self._reward_counter if state is None else self._reward_counter[tuple(state)]
        action_counter = self._action_counter if state is None else self._action_counter[tuple(state)]

        alpha = 1.0 + reward_counter
        beta = 1.0 + action_counter - reward_counter
        samples = np.random.beta(alpha, beta)
        action = np.argmax(samples)
        return action


if __name__ == '__main__':
    num_epochs = 100
    time_steps_per_epoch = 200
    env = MultiArmedBandit([0.8 * np.random.rand() for i in range(999)] + [0.95])
    # Create agents
    agents = [
        (RandomAgent(env), "Random"),
        (EpsilonGreedyAgent(env, epsilon=0.0), "Greedy"),
        (EpsilonGreedyAgent(env, epsilon=0.2), "EpsilonGreedy (e=0.2)"),
        (UCB1Agent(env), "UCB1 (c=sqrt(2))"),
        (UCB1Agent(env, c=0.2), "UCB1 (c=0.2)"),
        (ProbabilityMatching(env), "Prob. Match."),
    ]

    # Create plotting object
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 10))
    plt.suptitle("Unit: [Metric/epoch]")
    x = np.arange(num_epochs)
    for agent, agent_name in agents:
        epoch_mean_rewards, epoch_utility_estimation_rmse = agent.train(num_epochs=num_epochs, time_steps_per_epoch=time_steps_per_epoch)
        ax1.plot(x, epoch_mean_rewards, label=agent_name)
        ax2.plot(x, epoch_utility_estimation_rmse, label=agent_name)
        ax3.plot(x, np.max(env.utility) - epoch_mean_rewards, label=agent_name)

    ax1.title.set_text('Mean rewards')
    ax2.title.set_text('Utility estimate RMSE')
    ax3.title.set_text('Regret')
    ax1.set_ylim([0.0, 1.0])
    ax2.legend(loc='upper right')
    plt.show()
