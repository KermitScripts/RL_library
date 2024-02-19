import time
import gymnasium as gym
from tqdm import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F

from helper import episode_reward_plot
from ddqn import DDQN


class BCQNetwork(nn.Module):
    """
    Policy and imitation network for discrete BCQ algorithm
    """

    def __init__(self, num_obs, num_actions):
        super(BCQNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_features=num_obs, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_actions)
        )

        self.imitation = nn.Sequential(
            nn.Linear(in_features=num_obs, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_actions)
        )

    def forward(self, x):
        return self.layers(x), F.log_softmax(self.imitation(x), dim=1)


class BCQ:
    """
    Discrete BCQ algorithm following https://arxiv.org/abs/1910.01708
    """

    def __init__(self, env, gamma=0.99, learning_rate=3e-4, sync_after=5, batch_size=32, threshold=0.3,
                 val_freq=500, val_episodes=10):
        self.obs_dim, self.act_dim = env.observation_space.shape[0], env.action_space.n
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.sync_after = sync_after
        self.learning_rate = learning_rate
        self.val_freq = val_freq
        self.val_episodes = val_episodes

        self.Q_net = BCQNetwork(num_obs=self.obs_dim, num_actions=self.act_dim)
        self.Q_target_net = BCQNetwork(self.obs_dim, self.act_dim)
        self.Q_target_net.load_state_dict(self.Q_net.state_dict())

        self.optim = optim.Adam(self.Q_net.parameters(), lr=self.learning_rate)

        self.threshold = threshold

    def predict(self, state):
        """Predict the best action based on state using batch-constrained modification

        Returns
        -------
        int
            The action to be taken.
        """

        with torch.no_grad():
            # TODO: Select action from constrained set (using threshold)
            q_values, imts = self.Q_net.forward(state)
            imts = imts.exp()
            imts = (imts / imts.max(1, keepdim=True)[0] > self.threshold).float()
            actions = (imts * q_values + (1 - imts * -1e8)).argmax(1)

        return actions

    def train(self, offline_buffer, trainsteps):
        """
        Train the BCQ algorithm
        """

        val_rewards = []

        if self.batch_size > len(offline_buffer):
            raise RuntimeError('Not enough data in buffer!')

        print('Running BCQ on offline data..')
        time.sleep(1)
        for trainstep in tqdm(range(trainsteps)):

            obs, actions, rewards, next_obs, done = offline_buffer.get(self.batch_size)

            next_obs = torch.stack([torch.Tensor(next_ob) for next_ob in next_obs])
            next_actions = self.predict(next_obs)

            with torch.no_grad():
                next_q_values, _ = self.Q_target_net(next_obs)

                expected_q_values = torch.Tensor(rewards) + self.gamma * (1.0 - torch.Tensor(done)) * \
                                    next_q_values.gather(1, torch.LongTensor(next_actions).unsqueeze(1)).squeeze(1)

            obs = torch.stack([torch.Tensor(ob) for ob in obs])
            q_values, imt = self.Q_net(obs)
            q_values = q_values.gather(1, torch.LongTensor(actions).unsqueeze(1)).squeeze(1)

            q_loss = F.mse_loss(q_values, expected_q_values)
            i_loss = F.nll_loss(imt, torch.LongTensor(actions))

            Q_loss = q_loss + i_loss

            self.optim.zero_grad()
            Q_loss.backward()
            self.optim.step()

            if trainstep % self.sync_after == 0:
                self.Q_target_net.load_state_dict(self.Q_net.state_dict())

            if trainstep % self.val_freq == 0:
                rewards = self.validate()
                val_rewards.extend(rewards)
                episode_reward_plot(val_rewards, trainstep, window_size=7, step_size=1)

        return val_rewards

    def validate(self):
        """
        Validate with given policy for 10000 timesteps
        """

        collected_rewards = []
        for _ in range(self.val_episodes):
            obs, _ = self.env.reset()
            timestep_counter = 0
            while True:
                with torch.no_grad():
                    action = self.predict(torch.Tensor(obs).unsqueeze(0))
                obs, reward, terminated, truncated, _ = self.env.step(action.item())
                timestep_counter += 1
                if terminated or truncated:
                    collected_rewards.append(timestep_counter)
                    break
        return collected_rewards


if __name__ == '__main__':

    _env = gym.make('CartPole-v1')

    ddqn = DDQN(_env)
    res = ddqn.train(timesteps=30000)

    val = ddqn.validate()
    print('Average reward using policy for data generation: {:.1f}'.format(val))

    data = ddqn.generate_data(data_size=20000)

    bcq = BCQ(_env, threshold=0.3)
    bcq.train(data, trainsteps=50000)
